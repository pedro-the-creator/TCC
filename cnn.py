import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

imagePatches = glob('C:/Users/pedro/Downloads/archive/*/*/*')

imagePatches = [imagePatches[i] for i in range(len(imagePatches)) if 'IDC' not in imagePatches[i]]

y = []
for img in imagePatches:
    if img.endswith('class0.png'):
        y.append(0)
    elif img.endswith('class1.png'):
        y.append(1)

class MyDataset(Dataset):
    def __init__(self, df_data, transform=None):
        super().__init__()
        self.df = df_data.values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path, label = self.df[index]
        image = cv2.imread(img_path)
        image = cv2.resize(image, (50, 50))
        if self.transform is not None:
            image = self.transform(image)
        return image, label

images_df = pd.DataFrame()
images_df["images"] = imagePatches
images_df["labels"] = y

train, test = train_test_split(images_df, stratify=images_df.labels, test_size=0.2, random_state=42)
train, val = train_test_split(train, stratify=train.labels, test_size=0.2, random_state=42)

num_epochs = 20
num_classes = 2
batch_size = 32
learning_rate = 0.0001

trans_train = transforms.Compose([transforms.ToPILImage(),
                                  transforms.Pad(10, padding_mode='reflect'),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.RandomVerticalFlip(),
                                  transforms.RandomRotation(20),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

trans_valid = transforms.Compose([transforms.ToPILImage(),
                                  transforms.Pad(10, padding_mode='reflect'),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

dataset_train = MyDataset(df_data=train, transform=trans_train)
dataset_valid = MyDataset(df_data=val, transform=trans_valid)
dataset_test = MyDataset(df_data=test, transform=trans_valid)

loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
loader_valid = DataLoader(dataset=dataset_valid, batch_size=batch_size//2, shuffle=True, num_workers=0)
loader_test = DataLoader(dataset=dataset_test, batch_size=batch_size//2, shuffle=False, num_workers=0)

resnetmodel = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
num_ftrs = resnetmodel.fc.in_features
resnetmodel.fc = nn.Linear(num_ftrs, num_classes)

for n, p in resnetmodel.named_parameters():
    print(p.requires_grad)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnetmodel.parameters(), lr=learning_rate, momentum=0.9)

resnetmodel.to(device)

best_accuracy = 0
best_weights = None

trl = []
trac = []
vall = []
valac = []

print('start training')
for epoch in range(num_epochs):
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    resnetmodel.train()
    for i, (inputs, targets) in enumerate(loader_train):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = resnetmodel(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        train_loss += loss.item() * inputs.size(0)
        train_correct += (predicted == targets).sum().item()
        train_total += targets.size(0)

    train_loss /= len(train)
    train_acc = train_correct / train_total

    trl.append(train_loss)
    trac.append(train_acc)

    val_loss = 0.0
    val_correct = 0
    val_total = 0
    resnetmodel.eval()
    with torch.no_grad():
        for inputs, targets in loader_valid:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = resnetmodel(inputs)
            loss = criterion(outputs, targets)
            _, predicted = torch.max(outputs.data, 1)
            val_loss += loss.item() * inputs.size(0)
            val_correct += (predicted == targets).sum().item()
            val_total += targets.size(0)

    val_loss /= len(val)
    val_acc = val_correct / val_total

    vall.append(val_loss)
    valac.append(val_acc)

    if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_weights = resnetmodel.state_dict()

    print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

plt.plot(valac)
plt.plot(trac)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['validation', 'train'])

cuda_tensor = torch.tensor(vall)
vls = cuda_tensor.cpu()
cuda_tensor = torch.tensor(trl)
tls = cuda_tensor.cpu()

plt.plot(vls)
plt.plot(tls)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['validation', 'train'])

resnetmodel.load_state_dict(best_weights)
resnetmodel.to(device)

resnetpredict = []
resnetlabel = []

resnetmodel.eval()
confusion_matrix = torch.zeros(2, 2)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in loader_test:
        images = images.to(device)
        labels = labels.to(device)
        outputs = resnetmodel(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        resnetpredict.extend(predicted)
        resnetlabel.extend(labels)

        for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))

label_resnet = [tensor.cpu().numpy() for tensor in resnetlabel]
resnet_array = [tensor.cpu().numpy() for tensor in resnetpredict]

print(confusion_matrix(label_resnet, resnet_array))
print(classification_report(label_resnet, resnet_array))

dfv = pd.DataFrame()
dfv["resnet"] = resnet_array
dfv["label"] = label_resnet
dfv.head()
dfv.to_csv('resnetwithaug.csv')

torch.save({
    'model_state_dict': resnetmodel.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
}, 'checkpointresnet50withaug.pth')
