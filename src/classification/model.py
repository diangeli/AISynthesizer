import torch
import torch.nn as nn
import torchvision.models as models


class Classifier(nn.Module):
    def __init__(self, num_classes=91, encoder='simple', pretrained=False):
        super(Classifier, self).__init__()

        self.encoder_type = encoder

        if encoder == 'simple':
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 6, kernel_size=5),
                nn.ReLU(),
                nn.MaxPool2d(2, 2), 
                nn.Conv2d(6, 16, kernel_size=5),
                nn.ReLU(),
                nn.MaxPool2d(2, 2), 
                nn.Flatten()
            )            

            self. decoder = nn.Sequential(
                nn.Linear(16 * 53 * 53, 120),
                nn.ReLU(),
                nn.Linear(120, num_classes)
            )

        else:
            if encoder == 'resnet50':
                self.encoder = models.resnet50(weights='IMAGENET1K_V1')
            elif encoder == 'resnet18':
                self.encoder = models.resnet18(weights='IMAGENET1K_V1')
            elif encoder == 'vit16':
                self.encoder = models.vit_b_16(weights='IMAGENET1K_V1')

            self.decoder = nn.Sequential(
                nn.Linear(1000, 120),
                nn.ReLU(),
                nn.Linear(120, num_classes)
            )
        
        

    def forward(self, x):
        x = self.encoder(x)  # Apply first convolution, ReLU activation, and max pooling
        x = self.decoder(x) # Apply third fully connected layer (no activation)
        return x