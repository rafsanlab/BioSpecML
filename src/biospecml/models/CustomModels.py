import torch
import torch.nn as nn
import torchvision.models as models

class CustomModel(nn.Module):
    """

    Options and input image size:
        - "vgg16", "vgg19" : 
            any input size, default = (224, 224)
        - "inception_v3" :
            only (299, 299)
        - "densenet121", "densenet161","densenet169","densenet201" : 
            any input size, default = (224, 224)
    """

    def __init__(self, model_name, num_classes, hidden_units=None):
        super(CustomModel, self).__init__()

        # Select the model
        if model_name == "vgg16":
            self.model = models.vgg16(pretrained=True)
            num_features = self.model.classifier[6].in_features
            # Replace the classifier head
            self.model.classifier[6] = nn.Identity()
        elif model_name == "vgg19":
            self.model = models.vgg19(pretrained=True)
            num_features = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Identity()
        elif model_name == "inception_v3":
            self.model = models.inception_v3(pretrained=True, aux_logits=True)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Identity()
        elif model_name == "densenet121":
            self.model = models.densenet121(pretrained=True)
            num_features = self.model.classifier.in_features
            self.model.classifier = nn.Identity()
        elif model_name == "densenet169":
            self.model = models.densenet169(pretrained=True)
            num_features = self.model.classifier.in_features
            self.model.classifier = nn.Identity()
        elif model_name == "densenet201":
            self.model = models.densenet201(pretrained=True)
            num_features = self.model.classifier.in_features
            self.model.classifier = nn.Identity()
        elif model_name == "densenet161":
            self.model = models.densenet161(pretrained=True)
            num_features = self.model.classifier.in_features
            self.model.classifier = nn.Identity()
        else:
            raise ValueError(f"<model_name> accepted values: 'vgg16', 'vgg19', 'inception_v3', 'densenet121', 'densenet169', 'densenet201', or 'densenet161'.")

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
        x = self.model(x)  # Extract features from the selected model
        if isinstance(x, tuple):  # Handle tuple output for inception
            x = x[0]
        x = self.fc(x)  # Final classification layer
        return x
    