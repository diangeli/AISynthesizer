import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
import numpy as np
import mido
from mido import Message, MidiFile, MidiTrack
from model import ViViT 
from data import VideoDataset
from torchvision import transforms
from utils import create_midi


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ViViT(
    image_size=(224, 224),
    num_frames=125,
    num_classes=88,
    dim=1024,
    depth=6,
    heads=8,
    mlp_dim=2048,
    pool='mean',
    channels=3,
    dim_head=64,
    dropout=0.1,
    emb_dropout=0.1
).to(device)

# Load the trained weights
model.load_state_dict(torch.load('path_to_your_trained_model.pth'))
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Resize to the input size expected by the model
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# TODO: fix VideoDataset loader so that it takes only silent videos as an input and makes them ready for the model.
test_dataset = VideoDataset(video_dir='test_silent_videos', midi_dir='midis', transforms=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


with torch.no_grad():
    for idx, batch in enumerate(test_loader):
        video_tensors = batch['video'].to(device)
        probabilities = model(video_tensors).squeeze().cpu().numpy()

        output_file = f"midi_output_{idx}.mid"
        create_midi(probabilities, output_file=output_file, threshold=0.5)
        print(f"MIDI file saved to {output_file}")
