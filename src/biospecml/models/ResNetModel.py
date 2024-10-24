import torch
import torch.nn as nn
import torchvision.models as models

class ResNetModel(nn.Module):
    def __init__(self, num_classes, hidden_units=None, resnet=18):
        super(ResNetModel, self).__init__()
        
        # Select the ResNet model
        if resnet == 152:
            self.resnet = models.resnet152(pretrained=True)
        elif resnet == 101:
            self.resnet = models.resnet101(pretrained=True)
        elif resnet == 50:
            self.resnet = models.resnet50(pretrained=True)
        elif resnet == 34:
            self.resnet = models.resnet34(pretrained=True)
        elif resnet == 18:
            self.resnet = models.resnet18(pretrained=True)
        else:
            raise ValueError(f'<resnet> accepted values: 18, 34, 50, 101, or 152.')
        
        # Remove the fully connected layer of the pre-trained ResNet model
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Replace the fc layer with Identity to keep the features

        # Fully connected layers
        if hidden_units is not None:
            self.fc = nn.Sequential(
                nn.Linear(num_features, hidden_units),  # First FC layer
                nn.ReLU(),
                nn.Linear(hidden_units, num_classes)  # Output layer for num_classes
            )
        else:
            self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)  # Output shape: (batch_size, num_classes)
        
        return x