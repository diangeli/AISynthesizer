import os
from torch.utils.data import Dataset
from PIL import Image
import torch

class FramesDataset(Dataset):
    def __init__(self, frames_dir, class_num, transforms=None):
        self.frames_dir = frames_dir
        self.transforms = transforms
        self.class_num = class_num

        self.frames = [[os.path.join(frames_dir, file), file.split('.')[0].split('_')] for file in os.listdir(frames_dir)]
        
    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame_path = self.frames[idx][0]
        frame_tensor = self.transforms(self.load_frame(frame_path))
        labels = self.frames[idx][1]
        label_tensor = self.get_label(labels)
        return [frame_tensor, label_tensor]

    def load_frame(self, frame_path):
        frame = (Image.open(frame_path))
        return frame
    
    def get_label(self, label):
        tensor_label = torch.zeros(self.class_num)
        for i in range(len(label)):
            tensor_label[int(label[i])] = 1
        return tensor_label
