import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import time
from tqdm import tqdm

from sklearn.metrics import classification_report, f1_score, confusion_matrix

from architectures.cnn_ta import CNN
from architectures.vit import VisionTransformer
from architectures.sam.sam import SAM

import os


# 하이퍼파라미터 설정
batch_size = 128
num_epochs = 40
learning_rate = 0.001


# 데이터셋 불러오기
etf_list = ['KRW-BTC']
threshold = '01'

def load_dataset():
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for etf in etf_list:
        x_train.extend(np.load(f"ETF/strategy/{threshold}/TrainData/x_{etf}.npy"))
        y_train.extend(np.load(f"ETF/strategy/{threshold}/TrainData/y_{etf}.npy"))
        x_test.extend(np.load(f"ETF/strategy/{threshold}/TestData/x_{etf}.npy"))
        y_test.extend(np.load(f"ETF/strategy/{threshold}/TestData/y_{etf}.npy"))
    
    x_train_new = []
    y_train_new = []
    for x_t, y_t in zip(x_train, y_train):
        if y_t != 1:
            x_train_new.append(x_t)
            y_train_new.append(y_t)
            x_train_new.append(x_t)
            y_train_new.append(y_t)

    x_train.extend(x_train_new)
    y_train.extend(y_train_new)
    unique, counts = np.unique(y_train, return_counts=True)
    print(np.asarray((unique, counts)).T)
    
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    return x_train, y_train, x_test, y_test

def prepare_dataset(x_train, y_train, x_test):
    val_split = 0.1
    val_size = int(len(x_train) * val_split)
    train_size = len(x_train) - val_size

    train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    print(f"Training data samples: {len(train_dataset)}")
    print(f"Validation data samples: {len(val_dataset)}")
    print(f"Test data samples: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset

def get_dataloader(dataset, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_experiment(model, train_loader, val_loader, test_loader):
    base_optimizer = optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, lr = learning_rate, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for epoch in tqdm(range(num_epochs)):

        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.first_step(zero_grad=True)

            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.second_step(zero_grad=True)
            

            running_loss += loss.item()

        model.eval()
        val_loss = 0.0
        lowest_val_loss = 1000
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, Val Accuracy: {100 * correct / total:.2f}%")
        if val_loss < lowest_val_loss:
            lowest_val_loss = val_loss
            # if directory doesn't exists
            os.makedirs('./model_save', exist_ok=True)
            name = 'CNN' if isinstance(model, CNN) else 'ViT'
            torch.save(model.state_dict(), f"./model_save/{name}.pt")
            print('Model Saved')

    print("Finished Training")

    # 테스트 데이터셋 평가
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())
    cf = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:\n{cf}")
    cr = classification_report(y_true, y_pred)
    print(f"\nClassification Report:\n{cr}")
    f1 = f1_score(y_true, y_pred, average='micro')
    print(f"\nF1 Score: {f1}")


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_dataset()
    train_dataset, val_dataset, test_dataset = prepare_dataset(x_train, y_train, x_test)
    train_loader = get_dataloader(train_dataset, shuffle=True)
    val_loader = get_dataloader(val_dataset)
    test_loader = get_dataloader(test_dataset)

    # model = CNN()
    model = VisionTransformer()
    run_experiment(model, train_loader, val_loader, test_loader)
