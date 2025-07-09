""" By @rafsanlab at BioSpecML Github. """

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearNetBasicBlock(nn.Module):
    """
    A basic linear block inspired by the ResNet architecture.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        dropout_rate (float, optional): Dropout rate. Default is None.
        residual_mode (bool): If True, enables residual connection. Default is False.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout_rate: float = None,
        residual_mode: bool = False,
    ):
        super().__init__()
        self.residual_mode = residual_mode

        # self.block = nn.Sequential(
        #     nn.Linear(in_features, in_features),
        #     nn.LayerNorm(in_features),
        #     nn.ReLU(),
        #     nn.Linear(in_features, out_features),
        #     nn.LayerNorm(out_features),
        # )
        self.block = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features),
            nn.ReLU(),
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
        )
        self.activation_layer = nn.ReLU()
        self.dropout_layer = nn.Dropout(p=dropout_rate) if dropout_rate else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.residual_mode:
            identity = x
            x = self.block(x)
            if identity.shape[1] != x.shape[1]:
                identity = F.pad(identity, (0, x.shape[1] - identity.shape[1]))
            x = self.activation_layer(x + identity)
        else:
            x = self.activation_layer(self.block(x))

        if self.dropout_layer:
            x = self.dropout_layer(x)
        return x


class MILAttention(nn.Module):
    """
    MIL Attention mechanism to aggregate instance-level features.

    Args:
        input_dim (int): Input feature dimension.
        hidden_dim (int): Dimension for attention space.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.attention_V = nn.Linear(input_dim, hidden_dim)
        self.attention_w = nn.Linear(hidden_dim, 1)

    def forward(self, H: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        A = torch.tanh(self.attention_V(H))  # (B, N, H)
        A = self.attention_w(A).squeeze(-1)  # (B, N)
        attention_weights = F.softmax(A, dim=1)  # (B, N)
        M = torch.bmm(attention_weights.unsqueeze(1), H).squeeze(1)  # (B, D)
        return M, attention_weights


class LinearNet(nn.Module):
    """
    Configurable Linear Neural Network with optional residual blocks, attention, and weak supervision.

    Args:
        input_size (int): Input feature size.
        hidden_size (int): Hidden layer size.
        num_classes (int): Number of output classes.
        hidden_expansion (str): 'same', 'double', or 'half' to control expansion of hidden layers.
        dropout_rate (float, optional): Dropout rate. Default is None.
        residual_mode (bool): Whether to use residual connections.
        add_num_blocks (int): Number of additional blocks. Default is 1.
        weakly_sv (bool): If True, treats input as weakly supervised (e.g. [B, N, F]).
        mil_aggregation_mode (str): 'attention', 'mean' or None. Determines how instances are aggregated in MIL.
        use_softmax (bool): Whether to apply softmax at output.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        hidden_expansion: str = "same",
        dropout_rate: float = None,
        residual_mode: bool = False,
        add_num_blocks: int = 1,
        weakly_sv: bool = False,
        mil_aggregation_mode: str = None,  # Added: 'attention', 'mean', or None
        use_softmax: bool = True,
    ):
        super().__init__()
        self.residual_mode = residual_mode
        self.use_softmax = use_softmax
        self.weakly_sv = weakly_sv
        self.mil_aggregation_mode = mil_aggregation_mode

        if self.weakly_sv and self.mil_aggregation_mode not in ["attention", "mean"]:
            raise ValueError(
                "For weakly_sv mode, mil_aggregation_mode must be 'attention' or 'mean'."
            )

        self.first_layer = nn.Linear(input_size, hidden_size)
        self.basic_blocks = nn.ModuleList()
        current_size = hidden_size

        for _ in range(add_num_blocks):
            if hidden_expansion == "double":
                next_size = current_size * 2
            elif hidden_expansion == "half":
                next_size = max(1, current_size // 2)
            else:
                next_size = current_size

            self.basic_blocks.append(
                LinearNetBasicBlock(
                    current_size, next_size, dropout_rate, residual_mode
                )
            )
            current_size = next_size

        if self.mil_aggregation_mode == "attention":
            self.attention_layer = MILAttention(current_size, current_size * 2)

        self.classification_layer = nn.Linear(current_size, num_classes)

    def forward(self, x, return_attention=False):
        attention_weights = None

        if self.weakly_sv:
            B, N, F = x.shape
            x = x.view(B * N, F)  # Flatten for instance-level processing

        x = self.first_layer(x)

        for basic_block in self.basic_blocks:
            x = basic_block(x)

        if self.weakly_sv:
            x = x.view(B, N, -1)  # Reshape back to (B, N, F_processed)

            if self.mil_aggregation_mode == "attention":
                x, attention_weights = self.attention_layer(x)
            elif self.mil_aggregation_mode == "mean":
                x = torch.mean(x, dim=1)  # Aggregate by taking the mean across instances

        x = self.classification_layer(x)

        if self.use_softmax:
            x = F.softmax(x, dim=1)

        if return_attention:
            return x, attention_weights
        return x


# # visualise attentions
# model.eval()
# with torch.no_grad():
#     outputs, attention = model(tnsr, return_attention=True)  # inputs: [B, N, F]

#     for i in range(attention.shape[0]):
#         weights = attention[i].cpu().numpy()  # shape (N,)
#         with open(f"bag_{i}_attention.csv", "w") as f:
#             for j, w in enumerate(weights):
#                 f.write(f"instance_{j},{w:.4f}\n")

# import seaborn as sns
# sns.heatmap(attention.cpu().numpy(), cmap="viridis")
