import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearNetBasicBlock(nn.Module):
    """
    A Basic block for the linear net inspired by the ResNet block.

    Args:
    - in_features (int): Number of input features.
    - out_features (int): Number of output features.
    - dropout_rate (float, optional): Dropout rate for dropout layers. Default is None.
    """
    def __init__(self, in_features, out_features, dropout_rate=None):
        super(LinearNetBasicBlock, self).__init__()
        
        # layers in this block
        layers = [
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
        ]
        
        # add dropout layer if needed
        if dropout_rate != None: 
            layers.append(nn.Dropout(p=dropout_rate))
        
        # unpack layers
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class LinearNet(nn.Module):
    """
    A Linear NN that allows multiple blocks.

    Args:
    - input_size (int): Input node for the first layer.
    - hidden_size (int): Size of the hidden layers.
    - num_classes (int): Number of output classes.
    - dropout_rate (float, optional): Dropout rate for dropout layers. Default is None.
    - residual_mode (bool, optional): If True, enables residual connections between blocks. Default is False.
    - add_num_blocks (int, optional): Number of additional blocks to add. Default is 0.
    - use_softmax (bool, optional): If True, applies softmax activation to the output. Default is True.
    """
    def __init__(self, input_size:int, hidden_size:int, num_classes:int,
                 dropout_rate:float=None, residual_mode:bool=False,
                 add_num_blocks:int=0, use_softmax:bool=True):
        super(LinearNet, self).__init__()
        self.residual_mode = residual_mode
        self.use_softmax = use_softmax
        self.first_block = LinearNetBasicBlock(input_size, hidden_size, dropout_rate)
        self.match_dim_layer = nn.Linear(input_size, hidden_size)  
        self.basic_blocks = nn.ModuleList(
            [LinearNetBasicBlock(hidden_size, hidden_size, dropout_rate) for _ in range(add_num_blocks)]
            )
        self.add_num_blocks = add_num_blocks
        self.classification_layer = nn.Linear(hidden_size, num_classes)
    
    def __str__(self):
        layers = [
            f'First Block: {self.first_block}',
        ]
        layers += [f'Additional Block {i}: {block}' for i, block in enumerate(self.basic_blocks, 1)]
        if self.use_softmax:
            layers.append(f'Softmax Layer.')
        else:    
            layers.append(f'Classification Layer: {self.classification_layer}')
        return '\n'.join(layers)
    
    def forward(self, x):
        
        # check and apply residual connection in first block
        if self.residual_mode:
            # apply match layer so that input dim = first_block output dim
            identity = self.match_dim_layer(x)  
            x = self.first_block(x)
            x += identity
        else:
            x = self.first_block(x)

        # check and add additional blocks
        if self.add_num_blocks > 0:
            for block in self.basic_blocks:
                if self.residual_mode:
                    identity = x
                    x = block(x)
                    x += identity
                else:
                    x = block(x)

        # final layer and/or add softmax layer
        if self.use_softmax:
            x = F.softmax(x, dim=1) # using dim i; (batch, i)
        else:
            x = self.classification_layer(x)
        return x