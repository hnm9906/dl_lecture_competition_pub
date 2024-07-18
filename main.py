
import re
import random
import time
from statistics import mode
import os

from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from transformers import BertTokenizer, BertModel
from torch.cuda.amp import autocast, GradScaler

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def process_text(text):
    # 小文字に変換
    text = text.lower()

    # 冠詞の削除
    text = re.sub(r'\b(a|an|the)\b', '', text)

    # トークン化
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokens = tokenizer.tokenize(text)
    text = ' '.join(tokens)
    
    return text

# 1. データローダーの作成
class VQADataset(torch.utils.data.Dataset):
    def __init__(self, df_path, image_dir, transform=None, answer=True):
        self.transform = transform  # 画像の前処理
        self.image_dir = image_dir  # 画像ファイルのディレクトリ
        self.df = pd.read_json(df_path)  # 画像ファイルのパス，question, answerを持つDataFrame
        self.answer = answer

        # question / answerの辞書を作成
        self.question2idx = {}
        self.answer2idx = {}
        self.idx2question = {}
        self.idx2answer = {}

        # 質問文に含まれる単語を辞書に追加
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        for i, row in self.df.iterrows():
            question = process_text(row['question'])
            self.question2idx[question] = i
            self.idx2question[i] = question

            if self.answer and 'answer' in row:
                answer = row['answer']
                if answer not in self.answer2idx:
                    self.answer2idx[answer] = len(self.answer2idx)
                    self.idx2answer[self.answer2idx[answer]] = answer

    def update_dict(self, dataset):
        self.question2idx = dataset.question2idx
        self.answer2idx = dataset.answer2idx
        self.idx2question = dataset.idx2question
        self.idx2answer = dataset.idx2answer

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.df.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        question = self.df.iloc[idx, 1]
        question = process_text(question)
        question_tokens = self.tokenizer(question, return_tensors='pt', padding='max_length', max_length=20, truncation=True)
        
        if self.answer:
            answer = self.df.iloc[idx, 2]
            answer = self.answer2idx[answer]
            return image, question_tokens, answer

        return image, question_tokens

    def __len__(self):
        return len(self.df)

# データ拡張
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, padding=4),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 評価指標の実装
def VQA_criterion(batch_pred: torch.Tensor, batch_answers: torch.Tensor):
    total_acc = 0.

    for pred, answers in zip(batch_pred, batch_answers):
        acc = 0.
        for i in range(len(answers)):
            num_match = 0
            for j in range(len(answers)):
                if i == j:
                    continue
                if pred == answers[j]:
                    num_match += 1
            acc += min(num_match / 3, 1)
        total_acc += acc / 10

    return total_acc / len(batch_pred)

# モデルの定義
class VQAModel(nn.Module):
    def __init__(self, num_classes):
        super(VQAModel, self).__init__()
        self.vision_model = torchvision.models.resnet50(pretrained=True)
        self.vision_model.fc = nn.Identity()

        self.text_model = BertModel.from_pretrained('bert-base-uncased')

        self.classifier = nn.Linear(2048 + 768, num_classes)

    def forward(self, image, question_tokens):
        image_features = self.vision_model(image)
        text_features = self.text_model(**question_tokens).pooler_output

        combined_features = torch.cat((image_features, text_features), dim=1)
        output = self.classifier(combined_features)
        return output

# 学習の実装
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    scaler = GradScaler()  # Mixed Precision Trainingのためのスケーラー

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for image, question_tokens, answers in dataloader:
        image, question_tokens, answers = \
            image.to(device), {k: v.to(device) for k, v in question_tokens.items()}, answers.to(device)

        optimizer.zero_grad()
        
        with autocast():  # Mixed Precision Training
            pred = model(image, question_tokens)
            loss = criterion(pred, answers.squeeze())
        
        scaler.scale(loss).backward()  # Mixed Precision Training
        scaler.step(optimizer)  # Mixed Precision Training
        scaler.update()  # Mixed Precision Training

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == answers).float().mean().item()  # simple accuracy

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start

def eval(model, dataloader, criterion, device):
    model.eval()
    scaler = GradScaler()  # Mixed Precision Trainingのためのスケーラー

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    with torch.no_grad():
        for image, question_tokens, answers in dataloader:
            image, question_tokens, answers = \
                image.to(device), {k: v.to(device) for k, v in question_tokens.items()}, answers.to(device)
            
            with autocast():  # Mixed Precision Training
                pred = model(image, question_tokens)
                loss = criterion(pred, answers.squeeze())

            total_loss += loss.item()
            total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
            simple_acc += (pred.argmax(1) == answers).float().mean().item()  # simple accuracy

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start

def main():
    # deviceの設定
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 16

    # dataloader / model
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    train_dataset = VQADataset(df_path="./data/train.json", image_dir="./data/train", transform=transform)
    test_dataset = VQADataset(df_path="./data/valid.json", image_dir="./data/valid", transform=transform, answer=False)
    test_dataset.update_dict(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = VQAModel(num_classes=len(train_dataset.answer2idx)).to(device)

    # optimizer / criterion
    num_epoch = 5
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # train model
    for epoch in range(num_epoch):
        train_loss, train_acc, train_simple_acc, train_time = train(model, train_loader, optimizer, criterion, device)
        print(f"【{epoch + 1}/{num_epoch}】\n"
              f"train time: {train_time:.2f} [s]\n"
              f"train loss: {train_loss:.4f}\n"
              f"train acc: {train_acc:.4f}\n"
              f"train simple acc: {train_simple_acc:.4f}")

    # 提出用ファイルの作成
    model.eval()
    submission = []
    for image, question_tokens in test_loader:
        image, question_tokens = image.to(device), {k: v.to(device) for k, v in question_tokens.items()}
        pred = model(image, question_tokens)
        pred = pred.argmax(1).cpu().item()
        submission.append(pred)

    submission = [train_dataset.idx2answer[id] for id in submission]
    submission = np.array(submission)
    torch.save(model.state_dict(), "model.pth")
    np.save("submission.npy", submission)

if __name__ == "__main__":
    main()

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def process_text(text):
    # 小文字に変換
    text = text.lower()

    # 冠詞の削除
    text = re.sub(r'\b(a|an|the)\b', '', text)

    # トークン化
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokens = tokenizer.tokenize(text)
    text = ' '.join(tokens)
    
    return text

# 1. データローダーの作成
class VQADataset(torch.utils.data.Dataset):
    def __init__(self, df_path, image_dir, transform=None, answer=True):
        self.transform = transform  # 画像の前処理
        self.image_dir = image_dir  # 画像ファイルのディレクトリ
        self.df = pd.read_json(df_path)  # 画像ファイルのパス，question, answerを持つDataFrame
        self.answer = answer

        # question / answerの辞書を作成
        self.question2idx = {}
        self.answer2idx = {}
        self.idx2question = {}
        self.idx2answer = {}

        # 質問文に含まれる単語を辞書に追加
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        for i, row in self.df.iterrows():
            question = process_text(row['question'])
            self.question2idx[question] = i
            self.idx2question[i] = question

            if self.answer and 'answer' in row:
                answer = row['answer']
                if answer not in self.answer2idx:
                    self.answer2idx[answer] = len(self.answer2idx)
                    self.idx2answer[self.answer2idx[answer]] = answer

    def update_dict(self, dataset):
        self.question2idx = dataset.question2idx
        self.answer2idx = dataset.answer2idx
        self.idx2question = dataset.idx2question
        self.idx2answer = dataset.idx2answer

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.df.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        question = self.df.iloc[idx, 1]
        question = process_text(question)
        question_tokens = self.tokenizer(question, return_tensors='pt', padding='max_length', max_length=20, truncation=True)
        
        if self.answer:
            answer = self.df.iloc[idx, 2]
            answer = self.answer2idx[answer]
            return image, question_tokens, answer

        return image, question_tokens

    def __len__(self):
        return len(self.df)

# データ拡張
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, padding=4),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 評価指標の実装
def VQA_criterion(batch_pred: torch.Tensor, batch_answers: torch.Tensor):
    total_acc = 0.

    for pred, answers in zip(batch_pred, batch_answers):
        acc = 0.
        for i in range(len(answers)):
            num_match = 0
            for j in range(len(answers)):
                if i == j:
                    continue
                if pred == answers[j]:
                    num_match += 1
            acc += min(num_match / 3, 1)
        total_acc += acc / 10

    return total_acc / len(batch_pred)

# モデルの定義
class VQAModel(nn.Module):
    def __init__(self, num_classes):
        super(VQAModel, self).__init__()
        self.vision_model = torchvision.models.resnet50(pretrained=True)
        self.vision_model.fc = nn.Identity()

        self.text_model = BertModel.from_pretrained('bert-base-uncased')

        self.classifier = nn.Linear(2048 + 768, num_classes)

    def forward(self, image, question_tokens):
        image_features = self.vision_model(image)
        text_features = self.text_model(**question_tokens).pooler_output

        combined_features = torch.cat((image_features, text_features), dim=1)
        output = self.classifier(combined_features)
        return output

# 学習の実装
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    scaler = GradScaler()  # Mixed Precision Trainingのためのスケーラー

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for image, question_tokens, answers in dataloader:
        image, question_tokens, answers = \
            image.to(device), {k: v.to(device) for k, v in question_tokens.items()}, answers.to(device)

        optimizer.zero_grad()
        
        with autocast():  # Mixed Precision Training
            pred = model(image, question_tokens)
            loss = criterion(pred, answers.squeeze())
        
        scaler.scale(loss).backward()  # Mixed Precision Training
        scaler.step(optimizer)  # Mixed Precision Training
        scaler.update()  # Mixed Precision Training

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == answers).float().mean().item()  # simple accuracy

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start

def eval(model, dataloader, criterion, device):
    model.eval()
    scaler = GradScaler()  # Mixed Precision Trainingのためのスケーラー

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    with torch.no_grad():
        for image, question_tokens, answers in dataloader:
            image, question_tokens, answers = \
                image.to(device), {k: v.to(device) for k, v in question_tokens.items()}, answers.to(device)
            
            with autocast():  # Mixed Precision Training
                pred = model(image, question_tokens)
                loss = criterion(pred, answers.squeeze())

            total_loss += loss.item()
            total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
            simple_acc += (pred.argmax(1) == answers).float().mean().item()  # simple accuracy

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start

def main():
    # deviceの設定
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 1

    # dataloader / model
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    train_dataset = VQADataset(df_path="./data/train.json", image_dir="./data/train", transform=transform)
    test_dataset = VQADataset(df_path="./data/valid.json", image_dir="./data/valid", transform=transform, answer=False)
    test_dataset.update_dict(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = VQAModel(num_classes=len(train_dataset.answer2idx)).to(device)

    # optimizer / criterion
    num_epoch = 5
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # train model
    for epoch in range(num_epoch):
        train_loss, train_acc, train_simple_acc, train_time = train(model, train_loader, optimizer, criterion, device)
        print(f"【{epoch + 1}/{num_epoch}】\n"
              f"train time: {train_time:.2f} [s]\n"
              f"train loss: {train_loss:.4f}\n"
              f"train acc: {train_acc:.4f}\n"
              f"train simple acc: {train_simple_acc:.4f}")

    # 提出用ファイルの作成
    model.eval()
    submission = []
    for image, question_tokens in test_loader:
        image, question_tokens = image.to(device), {k: v.to(device) for k, v in question_tokens.items()}
        pred = model(image, question_tokens)
        pred = pred.argmax(1).cpu().item()
        submission.append(pred)

    submission = [train_dataset.idx2answer[id] for id in submission]
    submission = np.array(submission)
    torch.save(model.state_dict(), "model.pth")
    np.save("submission.npy", submission)

if __name__ == "__main__":
    main()
