import os

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from utils import Utils
from torchmetrics.classification import BinaryROC

from data.video_dataset import VideoDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@hydra.main(version_base=None, config_path="../config", config_name="config")
def predict_model(config: DictConfig):
    utils = Utils(config)
    model = utils.get_model().to(device)

    # Load the trained weights
    model.load_state_dict(torch.load(os.path.join(config.inference.models_path, config.inference.model_name)))
    model.eval()

    transform = utils.get_transforms()
    # Initialize the dataset for evaluation
    test_dataset = VideoDataset(video_dir='data/processed/videos/beethoven_test',
                                midi_dir="data/processed/midis/beethoven_test", num_frames= config.model.num_frames, transforms=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Use the DataLoader in your evaluation loop
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            print(f"Processing video {idx}")
            video_tensors = batch['video'].to(device)
            midis_tensors = batch['midi_labels'].to(device)
            probabilities = torch.sigmoid(model(video_tensors))
            metric = BinaryROC(thresholds=None)
            fpr , tpr, thresholds = metric(probabilities, midis_tensors.long())
            print(f"Best threshold = {thresholds[torch.argmax(tpr - fpr)]}")
            fig_, ax_ = metric.plot(score=True)
            fig_.savefig("data/output/roc_curve.png")
            metric = BinaryROC(thresholds=None).to(device)
            metric(probabilities, midis_tensors.long())
            fig_, ax_ = metric.plot(score=True)
            fig_.savefig("data/output/full_roc_curve.png")
            out_dir = "data/output/midis/beethoven_generated_midis"
            os.makedirs(out_dir, exist_ok=True)
            output_file = os.path.join(out_dir, f"generated_midi_{idx}.mid")
            notes = utils.process_roll(probabilities.squeeze().cpu().numpy())
            utils.generate_midi(notes, output_file)
            # utils.read_midi(output_file)
            # create_midi(probabilities, output_file, threshold=0.5, min_duration=480, max_notes_per_frame=20)
            
            
if __name__ == "__main__":
    
    predict_model()
