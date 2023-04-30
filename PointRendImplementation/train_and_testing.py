import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import Cityscapes
import torch
import torch.nn as nn
import torch.optim as optim
from model import PointRend

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("The device is {device}")

# Define transforms for data augmentation
transform_train = transforms.Compose([
    transforms.RandomCrop(512),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_val_test = transforms.Compose([
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Define dataset paths
train_set = Cityscapes(root='./VisualAid/', split='train', mode='fine', target_type='semantic', transform=transform_train)
print("Training Dataset has been completly loaded...")
test_set = Cityscapes(root='./VisualAid/', split='test', mode='fine', target_type='semantic', transform=transform_val_test)
print("Test Dataset has been completly loaded...")
val_set = Cityscapes(root='./VisualAid/', split='val', mode='fine', target_type='semantic', transform=transform_val_test)
print("Validation Dataset has been completly loaded...")


# Define data loaders
train_loader = data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=2)
print("Training DataLoader is complete.")
test_loader = data.DataLoader(test_set, batch_size=4, shuffle=True, num_workers=2)
print("Testing DataLoader is complete.")
val_loader = data.DataLoader(val_set, batch_size=4, shuffle=True, num_workers=2)
print("Validation DataLoader is complete.")

# Define model
model = PointRend(in_channels=3, out_channels=32, num_points=64).to(device)
print("Modle is loaded.")

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
print("Criterion is {criterion}")
optimizer = optim.Adam(model.parameters(), lr=0.001)
print("Optimizer is {optimizer}")

# Define training function
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(train_loader)
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, epoch_loss))

# Define validation function
def validate(model, device, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    epoch_loss = running_loss / len(val_loader)
    print('Validation Loss: {:.4f}'.format(epoch_loss))

# Define testing function
def test(model, device, test_loader):
    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            # Process outputs to calculate distance of objects
            # Write code to generate depth map from processed outputs

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    train(model, device, train_loader, optimizer, criterion, epoch)
    validate(model, device, val_loader, criterion)

# Test the model
test(model,device, val_loader)

