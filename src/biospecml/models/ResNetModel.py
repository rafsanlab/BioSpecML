import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class MILAttention(nn.Module):
    """ Basic MIL Attention (Ilse et al. 2018, Eq. 7) """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.attention_V = nn.Linear(input_dim, hidden_dim)
        self.attention_w = nn.Linear(hidden_dim, 1)

    def forward(self, H: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        A = torch.tanh(self.attention_V(H)) 
        A = self.attention_w(A).squeeze(-1) 
        attention_weights = F.softmax(A, dim=1) 
        M = torch.bmm(attention_weights.unsqueeze(1), H).squeeze(1) 
        return M, attention_weights

class GatedMILAttention(nn.Module):
    """ Gated MIL Attention (Ilse et al. 2018, Eq. 8) """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.attention_V = nn.Linear(input_dim, hidden_dim)
        self.attention_U = nn.Linear(input_dim, hidden_dim)
        self.attention_w = nn.Linear(hidden_dim, 1)

    def forward(self, H: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        A_V = torch.tanh(self.attention_V(H))
        A_U = torch.sigmoid(self.attention_U(H))
        A = self.attention_w(A_V * A_U).squeeze(-1)
        attention_weights = F.softmax(A, dim=1)
        M = torch.bmm(attention_weights.unsqueeze(1), H).squeeze(1)
        return M, attention_weights

class ResNetModel(nn.Module):
    def __init__(
        self, 
        num_classes, 
        hidden_units=None, 
        resnet=18, 
        pretrained=False, 
        weakly_sv=False, 
        use_attention=False,
        gated=False,         # New: Toggle Gated Attention
        in_channels=3        # New: Custom Image Channels
    ):
        super(ResNetModel, self).__init__()
        
        self.weakly_sv = weakly_sv
        self.use_attention = use_attention

        # 1. Select ResNet
        if resnet == 152:
            self.resnet = models.resnet152(pretrained=pretrained)
        elif resnet == 101:
            self.resnet = models.resnet101(pretrained=pretrained)
        elif resnet == 50:
            self.resnet = models.resnet50(pretrained=pretrained)
        elif resnet == 34:
            self.resnet = models.resnet34(pretrained=pretrained)
        elif resnet == 18:
            self.resnet = models.resnet18(pretrained=pretrained)
        else:
            raise ValueError(f'<resnet> accepted values: 18, 34, 50, 101, or 152.')
        
        # 2. Modify First Layer if in_channels != 3
        if in_channels != 3:
            # We must replace the first Conv2d layer. 
            # Original: Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.resnet.conv1 = nn.Conv2d(
                in_channels, 
                64, 
                kernel_size=7, 
                stride=2, 
                padding=3, 
                bias=False
            )
        
        # 3. Remove FC layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        # 4. Attention Mechanism
        if self.use_attention:
            hidden_dim = 128  # Adjust as needed
            if gated:
                self.attention_layer = GatedMILAttention(num_features, hidden_dim)
            else:
                self.attention_layer = MILAttention(num_features, hidden_dim)

        # 5. Classifier
        if hidden_units is not None:
            self.fc = nn.Sequential(
                nn.Linear(num_features, hidden_units),
                nn.ReLU(),
                nn.Linear(hidden_units, num_classes)
            )
        else:
            self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x, return_attention=False):
        attention_weights = None

        # Handle Weak Supervision (Batch, Bags, Channels, H, W)
        if self.weakly_sv:
            if x.dim() == 5:
                B, N, C, H, W = x.shape
                x = x.reshape(B * N, C, H, W) 
            else: 
                # Fallback / Safety
                B = x.shape[0]
                N = 1

        x = self.resnet(x) # (B*N, num_features)

        if self.weakly_sv:
            x = x.view(B, N, -1) # (B, N, num_features)

        if self.use_attention and self.weakly_sv:
            x, attention_weights = self.attention_layer(x) # (B, num_features)
        
        x = self.fc(x)
        
        if return_attention:
            return x, attention_weights
            
        return x
