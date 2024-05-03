from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from dataset import FramesDataset
from utils import save
from model import Classifier


def train(model, criterion, optimizer, epochs, train_loader, test_loader, device='cpu'):
    # Train the model
    best_model = model
    criterion = criterion
    optimizer = optimizer

    epochs = epochs
    best_val_loss = 1000
    test_loss_all = []
    train_loss_all = []
    accuracy_all = []
    for epoch in tqdm(range(epochs), unit='epoch'):  # Loop over the dataset multiple times

        model.train()
        train_loss = []
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            output = model(inputs)
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        model.eval()
        test_loss = []
        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in enumerate(test_loader, 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                output = model(inputs)
                test_loss.append(criterion(output, labels).cpu().item())

                outputs = torch.sigmoid(output).cpu()
                predicted = np.round(outputs)
                total += labels.size(0)*labels.size(1)
                correct += (predicted == labels.cpu()).sum().item()
                
        accuracy = 100*correct/total
        accuracy_all.append(accuracy)
        print(f'Epoch [{epoch+1}/{epochs}],Accuracy: {accuracy:.4f}%')

        train_loss = np.mean(train_loss)
        test_loss = np.mean(test_loss)
        train_loss_all.append(train_loss)
        test_loss_all.append(test_loss)

        if best_val_loss > test_loss:
            best_val_loss = test_loss
            best_model = model
            best_epoch = epoch
            print(f'Epoch [{epoch+1}/{epochs}],Test Loss: {train_loss:.8f}')
            print(f'Epoch [{epoch+1}/{epochs}],Test Loss: {test_loss:.8f}')

        
    print(f'Finished Training, best epoch: {best_epoch}')
    return best_model, train_loss_all, test_loss_all, accuracy_all


if __name__ == '__main__':
    # define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define transforms for the images
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    dataset = FramesDataset('note_frames_multi/', 91, transforms=transform)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    # dictionary with models 
    models_dict = {
        'simpleCNN': 'simple',
        'ResNet18': 'resnet18',
        'ResNet50': 'resnet50',
        'ViT16': 'vit16',
        'ViT32': 'vit32'
    }
    
    # Train models non pretrained
    for key in models_dict.keys():
        # Create an instance of the model
        model = Classifier(encoder=models_dict[key]).to(device)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        best_model, train_loss, test_loss, accuracy = train(model, criterion, optimizer, 100, train_loader, test_loader)
        save(key, best_model, train_loss, test_loss, accuracy)