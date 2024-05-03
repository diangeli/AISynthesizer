import os 
import numpy as np
import matplotlib.pyplot as plt

from torchmetrics.classification import MulticlassConfusionMatrix
from torch.utils.data import DataLoader
from torchvision import transforms
import torch

from dataset import FramesDataset
from model import Classifier
from tqdm import tqdm


def eval(model, val_loader, device='cpu'):
    total = 0
    correct = 0
    model.eval()
    all_target = []
    all_pred = []
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            output = model(inputs)

            outputs = torch.sigmoid(output).cpu()
            predicted = np.round(outputs)
            total += labels.size(0)*labels.size(1)
            correct += (predicted == labels.cpu()).sum().item()

            all_target.append(labels.to('cpu'))
            all_pred.append(predicted)

    all_target_np = np.concatenate((all_target[0], all_target[1]), axis=0)
    for target_id in range(2, len(all_target)):
        all_target_np = np.concatenate((all_target_np, all_target[target_id]), axis=0)

    all_pred_np = np.concatenate((all_pred[0], all_pred[1]), axis=0)
    for target_id in range(2, len(all_target)):
        all_pred_np = np.concatenate((all_pred_np, all_pred[target_id]), axis=0)


    metric = MulticlassConfusionMatrix(num_classes=2)
    metric(torch.tensor(all_pred_np[:, 1:]), torch.tensor(all_target_np[:, 1:]))
    return metric



if __name__=="__main__":
    # define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)

    # Load data
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    

    dataset = FramesDataset('note_frames_eval/', 91, transforms=transform)
    val_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    print("Loaded data.")

    conf_all = []
    for model_name in tqdm(os.listdir(os.path.join('results'))):
        print("Evaluating:", model_name)
        if 'simple' in model_name.lower():
            model = Classifier(encoder='simple')
            model.load_state_dict(torch.load(os.path.join('results', 'simpleCNN', 'model.pt')))
            model.to(device)

            conf = eval(model, val_loader, device)
            conf_all.append(['simple', conf])
        else: continue
        
        model_path = os.path.join('results', model_name, 'model.pt')
        pretrained = '_' in model_name
        
        model = Classifier(encoder=model_name.split('_')[0].lower(), pretrained=pretrained)
        model.load_state_dict(torch.load(os.path.join('results', model_name, 'model.pt')))
        model.to(device)

        conf = eval(model, val_loader, device)
        print(conf.compute())
        conf_all.append([model_name, conf])

    print(conf_all) 