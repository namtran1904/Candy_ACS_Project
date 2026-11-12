import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

#Choose device
if(torch.cuda.is_available()):
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

#Hyperparameters
BATCH_SIZE = 32
IMAGE_SIZE = 256
EPOCHS = 10
LEARNING_RATE = 0.001
TRAIN_DIR = os.path.join("data", "data_set", "train_data_set") 
TEST_DIR = os.path.join("data", "data_set", "test_data_set")

#Transforms raw data to train set
train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees = 30),
    transforms.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
 #transforms raw data to test set
test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#create 2 data sets
train_dataset = datasets.ImageFolder(root = TRAIN_DIR,transform = train_transforms)
test_dataset = datasets.ImageFolder(root = TEST_DIR, transform = test_transforms)

train_loader = DataLoader(train_dataset,batch_size = BATCH_SIZE, shuffle = True);
test_loader = DataLoader(test_dataset,batch_size = BATCH_SIZE, shuffle = False);

print(f"Device: {device}")
for class_name in train_dataset.classes:
    print(f"Class names: {class_name}")
    print(f"Batch size: {BATCH_SIZE}, Image size: {IMAGE_SIZE}, Epochs: {EPOCHS}, Learning rate: {LEARNING_RATE}\n")

num_classes = len(train_dataset.classes)

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

model.fc = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(model.fc.in_features,num_classes)
)

model = model.to(device)

#Hàm lost function
criterion = nn.CrossEntropyLoss()

#Hàm optimizer
optimizer = optim.Adam(model.parameters(),lr = LEARNING_RATE,weight_decay = 1e-4)

list_loss = []
list_accuracy = []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/(EPOCHS)]", leave = True) 
    for images, labels in loop:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        running_loss+= loss.item() * images.size(0)

        loop.set_postfix(los = loss.item())

    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch[{epoch+1}/{EPOCHS}] - loss: {epoch_loss:.4f}")
    list_loss.append(epoch_loss)

model.eval()
with torch.no_grad():
    torch.max(outputs.data,1)

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in test_loader:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _,predicted = torch.max(outputs,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy of the model on the test set: {100* correct/total: .2f}%")

#Vẽ đồ thị Loss
plt.style.use('seaborn-v0_8-whitegrid')  
plt.figure(figsize=(10,5))
x_axis = range(1,EPOCHS+1)
y_axis = list_loss
loss_plot = plt.plot(x_axis,list_loss,label = 'Training Loss', color = 'Orange',linewidth = 1.5)
plt.title('Training Loss Over Epochs')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.show()
    
