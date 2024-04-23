import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
import numpy as np
import mido
from mido import Message, MidiFile, MidiTrack
from model import ViViT 
from dataset import VideoDataset
from torchvision import transforms
from utils import create_midi, process_roll, generate_midi, read_midi
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ViViT(
    image_size=(224, 224),
    num_frames=250,
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
models_dir = "models"
model.load_state_dict(torch.load(os.path.join(models_dir, "10_epochs_new_model_beethoven.pth")))
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Resize to the input size expected by the model
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialize the dataset for evaluation
test_dataset = VideoDataset(video_dir='data/processed/videos/beethoven_test', transforms=transform, mode='eval')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Use the DataLoader in your evaluation loop
with torch.no_grad():
    for idx, batch in enumerate(test_loader):
        video_tensors = batch['video'].to(device)
        probabilities = model(video_tensors).squeeze().cpu().numpy()
        out_dir = "data/output/midis/beethoven_generated_midis"
        os.makedirs(out_dir, exist_ok=True)
        output_file = os.path.join(out_dir, "generated_midi.mid")
        notes = process_roll(probabilities)
        generate_midi(notes, output_file)
        read_midi(output_file)
        # create_midi(probabilities, output_file, threshold=0.5, min_duration=480, max_notes_per_frame=20)
