
# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES
# TO THE CORRECT LOCATION (/kaggle/input) IN YOUR NOTEBOOK,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.

import os
import sys
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError
from zipfile import ZipFile
import tarfile
import shutil
import subprocess

CHUNK_SIZE = 40960
DATA_SOURCE_MAPPING = 'breast-histopathology-images:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-data-sets%2F7415%2F10564%2Fbundle%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com%252F20240603%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20240603T170015Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D382f4a7d14b725e9663100926af9455c4af4343a55b7c7a91326a19cdf6247260d29a1374e75e8c4a9f3c4195ab26c4aa6c58495a66da7d73d12157f1ba0a1158d725b6b983d52397446e38204bb9ad31e882008021d055f9f0818521ef7fb8f47902d1df11fb7c47b07cdb0964de3e1e0215501b6d1d581282065332949439b1c2253114eaa32c257d372cce48d1fafcd7ec3b46217fa4f12814efea3b674464d621ba911a5fef4637bb218cf0719db880e024c61aa5a4eea2241f53dd9a1386adb375783b15a73ab78d65a7b11dcf2d8fba239b499671d00b754c7690955b929e9de3daca3f3a54a5936882eefabb9f38c520557be8dafe22211358197cd20'

KAGGLE_INPUT_PATH='/kaggle/input'
KAGGLE_WORKING_PATH='/kaggle/working'
KAGGLE_SYMLINK='kaggle'

command = "umount /kaggle/input/ 2> /dev/null"
shutil.rmtree('/kaggle/input', ignore_errors=True)
os.makedirs(KAGGLE_INPUT_PATH, 0o777, exist_ok=True)
os.makedirs(KAGGLE_WORKING_PATH, 0o777, exist_ok=True)

try:
  os.symlink(KAGGLE_INPUT_PATH, os.path.join("..", 'input'), target_is_directory=True)
except FileExistsError:
  pass
try:
  os.symlink(KAGGLE_WORKING_PATH, os.path.join("..", 'working'), target_is_directory=True)
except FileExistsError:
  pass

for data_source_mapping in DATA_SOURCE_MAPPING.split(','):
    directory, download_url_encoded = data_source_mapping.split(':')
    download_url = unquote(download_url_encoded)
    filename = urlparse(download_url).path
    destination_path = os.path.join(KAGGLE_INPUT_PATH, directory)
    try:
        with urlopen(download_url) as fileres, NamedTemporaryFile() as tfile:
            total_length = fileres.headers['content-length']
            print(f'Downloading {directory}, {total_length} bytes compressed')
            dl = 0
            data = fileres.read(CHUNK_SIZE)
            while len(data) > 0:
                dl += len(data)
                tfile.write(data)
                done = int(50 * dl / int(total_length))
                sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {dl} bytes downloaded")
                sys.stdout.flush()
                data = fileres.read(CHUNK_SIZE)
            if filename.endswith('.zip'):
              with ZipFile(tfile) as zfile:
                zfile.extractall(destination_path)
            else:
              with tarfile.open(tfile.name) as tarfile:
                tarfile.extractall(destination_path)
            print(f'\nDownloaded and uncompressed: {directory}')
    except HTTPError as e:
        print(f'Failed to load (likely expired) {download_url} to path {destination_path}')
        continue
    except OSError as e:
        print(f'Failed to load {download_url} to path {destination_path}')
        continue

print('Data source import complete.')

import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from glob import glob
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
from sklearn.svm import SVC
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device

imagePatches = glob('/kaggle/input/breast-histopathology-images/*/*/*')

len(imagePatches)

imagePatches = [imagePatches[i] for i in range(len(imagePatches)) if 'IDC' not in imagePatches[i]]

len(imagePatches)

y = []
for img in imagePatches:
    if img.endswith('class0.png'):
        y.append(0)
    elif img.endswith('class1.png'):
        y.append(1)

print(len(y))

class MyDataset(Dataset):
    def __init__(self, df_data,transform=None):
        super().__init__()
        self.df = df_data.values

        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path,label = self.df[index]

        image = cv2.imread(img_path)
        image = cv2.resize(image, (50,50))
        if self.transform is not None:
            image = self.transform(image)
        return image, label
    
images_df = pd.DataFrame()
images_df["images"] = imagePatches
images_df["labels"] = y
images_df.head()

train, test = train_test_split(images_df, stratify=images_df.labels, test_size=0.2,random_state=42)
train, val = train_test_split(train, stratify=train.labels, test_size=0.2,random_state=42)
len(train), len(val),len(test)

num_epochs = 20
num_classes = 2
batch_size = 128
learning_rate = 0.0001

trans_train = transforms.Compose([transforms.ToPILImage(),
                                  transforms.Pad(10, padding_mode='reflect'),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.RandomVerticalFlip(),
                                  transforms.RandomRotation(20),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])

trans_valid = transforms.Compose([transforms.ToPILImage(),
                                  transforms.Pad(10, padding_mode='reflect'),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])

dataset_train = MyDataset(df_data=train, transform=trans_train)
dataset_valid = MyDataset(df_data=val,transform=trans_valid)
dataset_test = MyDataset(df_data=test,transform=trans_valid)

loader_train = DataLoader(dataset = dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
loader_valid = DataLoader(dataset = dataset_valid, batch_size=batch_size//2, shuffle=True, num_workers=0)
loader_test = DataLoader(dataset = dataset_test, batch_size=batch_size//2, shuffle=False, num_workers=0)

vggmodel = models.vgg16(weights='IMAGENET1K_V1')
vggmodel.classifier[6] = nn.Linear(4096, num_classes)

for n, p in vggmodel.named_parameters():
    print(p.requires_grad)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(vggmodel.parameters(), lr=learning_rate, momentum=0.9)

vggmodel.to(device)

vgg_best_accuracy = 0
vgg_best_weights = None

trl = []
trac = []
vall = []
valac = []

for epoch in range(num_epochs):
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    vggmodel.train()
    for i, (inputs, targets) in enumerate(loader_train):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = vggmodel(inputs)
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
    vggmodel.eval()
    with torch.no_grad():
        for inputs, targets in loader_valid:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = vggmodel(inputs)
            loss = criterion(outputs, targets)
            _, predicted = torch.max(outputs.data, 1)
            val_loss += loss.item() * inputs.size(0)
            val_correct += (predicted == targets).sum().item()
            val_total += targets.size(0)

    val_loss /= len(val)
    val_acc = val_correct / val_total

    vall.append(val_loss)
    valac.append(val_acc)

    if val_acc > vgg_best_accuracy:
            vgg_best_accuracy = val_acc
            vgg_best_weights = vggmodel.state_dict()

    print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

plt.plot(valac)
plt.plot(trac)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['validation','train'])

cuda_tensor = torch.tensor(vall)
vls = cuda_tensor.cpu()
cuda_tensor = torch.tensor(trl)
tls = cuda_tensor.cpu()

plt.plot(vls)
plt.plot(tls)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['validation','train'])

vggmodel.load_state_dict(vgg_best_weights)
vggmodel.to(device)

vggpredict = []
vgglabel = []

vggmodel.eval()
confusion_matrix = torch.zeros(2, 2)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in loader_test:
        images = images.to(device)
        labels = labels.to(device)
        outputs = vggmodel(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        vggpredict.extend(predicted)
        vgglabel.extend(labels)

        for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))

label_vgg = [tensor.cpu().numpy() for tensor in vgglabel]
vgg_array = [tensor.cpu().numpy() for tensor in vggpredict]

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(label_vgg, vgg_array))
print(classification_report(label_vgg, vgg_array))

dfv = pd.DataFrame()
dfv["vgg"] = vgg_array
dfv["label"] = label_vgg
dfv.head()
dfv.to_csv('vgwithaug.csv')

torch.save({
    'model_state_dict': vggmodel.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
}, 'checkpointvgg50withaug.pth')