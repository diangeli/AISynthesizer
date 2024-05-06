import logging
import os

import hydra
import torch
from metrics import Metrics
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import Utils
import wandb
import torch.optim.lr_scheduler as lr_scheduler

from data.video_dataset import VideoDataset


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def train_model(config: DictConfig):
    utils = Utils(config)
    
    
    # wandb
    wandb_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=wandb_key)
    print(config)
    run = wandb.init(
        project="AISynthesizer", config=OmegaConf.to_container(config)
    )

    torch.manual_seed(config.training.seed)
    model = utils.get_model()

    epochs = config.training.epochs

    train_dataset = VideoDataset(video_dir=config.data.videos_training_path,
                                 midi_dir=config.data.midis_training_path,
                                 num_frames=config.model.num_frames, 
                                 transforms=utils.get_transforms())
    train_loader = DataLoader(train_dataset, batch_size=config.data.batch_size, shuffle=True, num_workers=config.data.num_workers)
    
    logger.info(f" - Train dataset: {len(train_loader)}")
    

    val_dataset = VideoDataset(video_dir=config.data.videos_val_path,
                                 midi_dir=config.data.midis_val_path,
                                 num_frames=config.model.num_frames, 
                                 transforms=utils.get_transforms())
    val_loader = DataLoader(val_dataset, batch_size=config.data.batch_size, shuffle=True, num_workers=config.data.num_workers)
    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define your loss function and optimizer
    loss_function = utils.get_loss_function()  # Or any other appropriate loss function
    optimizer = utils.get_optimizer(model)
    
    threshold = config.model.threshold
    
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

    # # Training loop
    for epoch in tqdm(range(epochs), desc="Epoch"):
        train_avg_loss = 0.0
        train_accuracy = 0.0
        train_precision = 0.0
        train_auroc = 0.0
        
        model.train()  # train mode
        for batch in tqdm(train_loader, desc="Batches"):
            video_tensors = batch["video"]  # Ensure this matches the dictionary key used in __getitem__
            midi_labels = batch["midi_labels"]
            video_tensors = video_tensors.to(device)
            midi_labels = midi_labels.to(device)
            
            # print(f"video: {video_tensors.shape}")
            # print(f"labels: {midi_labels.shape}")
            # Forward pass
            outputs = model(video_tensors)

            # Compute loss
            loss = loss_function(outputs, midi_labels)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            sig_outputs = torch.sigmoid(outputs)

            # print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
            # accuracy = ((outputs > threshold) == midi_labels).sum() / (midi_labels.shape[0] * midi_labels.shape[1])
            # print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], accuracy 0.4: {accuracy.item():.4f}')
            # print(f"midi_lables_sum = {midi_labels.sum().item()}")
            # print(f"out_labels_sum = {((outputs >= threshold)).sum().item()}")
            # print(outputs.shape)
            # accuracy = ((outputs > 0.1) == midi_labels).sum() / (midi_labels.shape[0] * midi_labels.shape[1])
            # print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], accuracy 0.1: {accuracy.item():.4f}')
            # print(f"out_labels_sum = {((outputs >= 0.1)).sum().item()}")
            
            train_avg_loss += loss.item() / len(train_loader)

            train_accuracy += Metrics.get_accuracy(outputs, midi_labels, threshold) / len(
                train_loader
            )
            train_precision += Metrics.get_precision(outputs, midi_labels, threshold) / len(
                train_loader
            )
            train_auroc += Metrics.get_auroc(outputs, midi_labels.long()) / len(train_loader)
        
        logger.info(
            f" - Training loss: {train_avg_loss}  - Training accuracy: {train_accuracy}"
        )
        logger.info(f" - Train specificity: {train_precision}")
        logger.info(f" - Train AUROC: {train_auroc}")

        wandb.log(
            {
                "train_loss": train_avg_loss,
                "train_accuracy": train_accuracy,
                "train_precision": train_precision,
                "train_auroc": train_auroc,
            }
        )
        
         ################################################################
        # Validation
        ################################################################

        # Compute the evaluation set loss
        val_avg_loss = 0.0
        val_accuracy = 0.0
        val_precision = 0.0
        val_auroc = 0.0

        model.eval()

        for batch in tqdm(
            val_loader, desc="Validation", leave=None
        ):
            video_tensors = batch["video"]  # Ensure this matches the dictionary key used in __getitem__
            midi_labels = batch["midi_labels"]
        
            video_tensors = video_tensors.to(device)
            midi_labels = midi_labels.to(device)
            
            with torch.no_grad():
                val_outputs = model(video_tensors)

            loss = loss_function(val_outputs, midi_labels)

            val_avg_loss += loss.item() / len(val_loader)

            val_accuracy += Metrics.get_accuracy(val_outputs, midi_labels, threshold) / len(
                val_loader
            )
            val_precision += Metrics.get_precision(val_outputs, midi_labels, threshold) / len(
                val_loader
            )
            val_auroc += Metrics.get_auroc(val_outputs, midi_labels.long()) / len(val_loader)
        
        logger.info(
            f" - Validation loss: {val_avg_loss}  - Validation accuracy: {val_accuracy}"
        )
        logger.info(f" - Validation specificity: {val_precision}")
        logger.info(f" - Validation AUROC: {val_auroc}")

        wandb.log(
            {
                "val_loss": val_avg_loss,
                "val_accuracy": val_accuracy,
                "val_precision": val_precision,
                "val_auroc": val_auroc,
            }
        )
        
        scheduler.step(val_avg_loss)

        # Save your model
        if config.model.save_model: 
            if epoch % 10 == 0 and epoch > 0:
                models_dir = "models"
                os.makedirs(models_dir, exist_ok=True)
                torch.save(
                    model.state_dict(),
                    os.path.join(config.model.save_path, utils.create_models_name(epoch) + ".pth"),
            )
            
    run.finish()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    train_model()
