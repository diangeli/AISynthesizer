import torch
import torch.nn as nn
import torchvision.models as models


class Classifier(nn.Module):
    def __init__(self, num_classes=91, encoder='simple', pretrained=False):
        super(Classifier, self).__init__()

        self.encoder_type = encoder
        if pretrained:
            self.weights = 'IMAGENET1K_V1'
        else: 
            self.weights = None

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
                self.encoder = models.resnet50(weights=self.weights)
            elif encoder == 'resnet18':
                self.encoder = models.resnet18(weights=self.weights)
            elif encoder == 'vit16':
                self.encoder = models.vit_b_16(weights=self.weights)
            elif encoder == 'vit32':
                self.encoder = models.vit_b_32(weights=self.weights)

            self.decoder = nn.Sequential(
                nn.Linear(1000, 120),
                nn.ReLU(),
                nn.Linear(120, num_classes)
            )
        

    def forward(self, x):
        x = self.encoder(x)  # Apply first convolution, ReLU activation, and max pooling
        x = self.decoder(x) # Apply third fully connected layer (no activation)
        return x