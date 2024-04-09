import torch
from omegaconf import DictConfig
from torchvision import transforms

from aisynthesizer.models.vivit import ViT


class Utils:
    def __init__(self, config: DictConfig):
        self.config = config
        self.transforms = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),  # Resize to the input size expected by the model
                transforms.ToTensor(),  # Convert images to PyTorch tensors
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        if config.data.enable_transforms is False:
            self.train_transforms = self.test_transforms

    def get_transforms(self):
        return self.transforms

    def get_model(self):
        if self.config.model.name == "vivit":
            return ViT(
                image_size=(224, 224),  # Height and width of input frames
                num_frames=125,  # Total number of frames in each video
                num_classes=88,  # For example, 88 keys on the piano
                dim=1024,  # Dimensionality of the token/patch embeddings
                depth=6,  # Number of transformer blocks (depth)
                heads=8,  # Number of attention heads
                mlp_dim=2048,  # Dimensionality of the feedforward layer
                pool="cls",  # Pooling method ('cls' for class token, 'mean' for mean pooling)
                channels=3,  # Number of channels in the video frames (RGB, so 3)
                dim_head=64,  # Dimensionality of each attention head
                dropout=0.1,  # Dropout rate
                emb_dropout=0.1,  # Embedding dropout rate
            )
        else:
            raise NotImplementedError(f"{self.config.model.name} model not yet supported!")

    def get_optimizer(self, model):
        if self.config.training.optimizer == "adam":
            return torch.optim.Adam(model.parameters(), lr=self.config.training.lr)
        else:
            raise NotImplementedError(f"{self.config.training.optimizer} optimizer not yet supported!")

    # def get_loss_function(self):
    #     if self.config.model.loss == "ce+":
    #         return nn.CrossEntropyLoss()
    #     elif self.config.model.loss == "dice":
    #         return DiceLoss('multiclass')
    #     elif self.config.model.loss == "focal":
    #         return FocalLoss('multiclass')
    #     else:
    #         return TverskyLoss('multiclass')

    def create_models_name(self):
        return f"{self.config.model.name}_{self.config.training.optimizer}_\
                 {self.config.training.lr}_{self.config.training.epochs}"
