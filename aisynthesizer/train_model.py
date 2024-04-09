import logging
import os

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import Utils

from data import VideoDataset

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def train_model(config: DictConfig):
    utils = Utils(config)
    model = utils.get_model()

    epochs = config.training.epochs

    train_dataset = VideoDataset(video_dir="silent_videos", midi_dir="midis", transforms=utils.get_transforms())
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=config.data.num_workers)

    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define your loss function and optimizer
    loss_function = torch.nn.BCEWithLogitsLoss()  # Or any other appropriate loss function
    optimizer = utils.get_optimizer(model)

    # # Training loop
    for epoch in tqdm(range(epochs), desc="Epoch"):
        for batch in train_loader:
            video_tensors = batch["video"]  # Ensure this matches the dictionary key used in __getitem__
            midi_labels = batch["midi_labels"]
            # For debugging
            print("Input video tensor shape:", video_tensors.shape)

            print(f"Type of video before calling .to(device): {type(video_tensors)}")
            video_tensors = video_tensors.to(device)
            midi_labels = midi_labels.to(device)

            # Forward pass
            outputs = model(video_tensors)

            # TODO: for each frame take the class with the biggest probability
            # TODO: fix training loop
            probabilities = outputs.detach().cpu().numpy()
            first_class_probabilities = probabilities[0, :, 0]
            frame_index = 0  # For the first frame
            frame_probabilities = probabilities[0, frame_index, :].tolist()

            print("Probabilities for the first class across all frames in the first video:", first_class_probabilities)
            print("Probabilities for all classes in the first frame of the first video:", frame_probabilities)

            # Compute loss
            loss = loss_function(outputs, midi_labels)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print loss (or store for later analysis)
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")
            print("outputs: ", outputs)

    # Save your model
    if config.model.save_model:
        torch.save(
            model.state_dict(),
            os.path.join(config.model.save_path, utils.create_models_name() + ".pth"),
        )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    train_model()
