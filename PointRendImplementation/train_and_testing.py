import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import Cityscapes
import torch
import torch.nn as nn
import torch.optim as optim
from model import PointRend
import numpy as np
# Define device
root_filename = '../dataVisualAid'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("The device is {device}")

def target_to_tensor(target):
    target = np.array(target).astype(np.int64)
    return torch.from_numpy(target)

target_transform = transforms.Compose([
    transforms.Resize(size=(512, 1024)),
    transforms.Lambda(lambda target: target_to_tensor(target)),
])

# Define transforms for data augmentation
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_val_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Define dataset paths
train_set = Cityscapes(root=root_filename, split='train', mode='fine', target_type='semantic', target_transform = target_transform,transform=transform_train)
print("Training Dataset has been loaded...")
test_set = Cityscapes(root=root_filename, split='test', mode='fine', target_type='semantic',target_transform = target_transform, transform=transform_val_test)
print("Test Dataset has been loaded...")
val_set = Cityscapes(root=root_filename, split='val', mode='fine', target_type='semantic',target_transform = target_transform, transform=transform_val_test)
print("Validation Dataset has been loaded...")


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
print(f"Criterion is {criterion}")
optimizer = optim.Adam(model.parameters(), lr=0.001)
print(f"Optimizer is {optimizer}")

# Define training function
def train(model, device, train_loader, optimizer, criterion, epoch,verbose=True):
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
        if verbose:
            print('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))
    epoch_loss = running_loss / len(train_loader)
    if verbose:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, epoch_loss))
    torch.save(model.state_dict(), 'model_epoch_{}.pt'.format(epoch+1))

# Define validation function
def validate(model, device, val_loader, criterion,verbose=True):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            if verbose:
                print('Validation Batch [{}/{}], Loss: {:.4f}'.format(i+1, len(val_loader), loss.item()))
    epoch_loss = running_loss / len(val_loader)
    if verbose:
        print('Validation Loss: {:.4f}'.format(epoch_loss))
    torch.save(model.state_dict(), 'model_epoch_{}_val.pt'.format(epoch+1))

# Define testing function
def test(model, device, test_loader,verbose=True):
    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if verbose:
                print('Test Batch [{}/{}]'.format(i+1, len(test_loader)))
            # Process outputs to calculate distance of objects
            # Write code to generate depth map from processed outputs

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    train(model, device, train_loader, optimizer, criterion, epoch)
    validate(model, device, val_loader, criterion)

# Test the model
test(model,device, test_loader)

