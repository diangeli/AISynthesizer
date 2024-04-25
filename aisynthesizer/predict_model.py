import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
import numpy as np
import mido
from utils import Utils
from omegaconf import DictConfig, OmegaConf
from data import VideoDataset
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@hydra.main(version_base=None, config_path="../config", config_name="config")
def predict_model(config: DictConfig):
    utils = Utils(config)
    model = utils.get_model().to(device)

    # Load the trained weights
    model.load_state_dict(torch.load(os.path.join(config.infererence.model_path, config.infererence.model_name)))
    model.eval()

    transform = utils.get_transforms()
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
            notes = utils.process_roll(probabilities)
            utils.generate_midi(notes, output_file)
            utils.read_midi(output_file)
            # create_midi(probabilities, output_file, threshold=0.5, min_duration=480, max_notes_per_frame=20)
