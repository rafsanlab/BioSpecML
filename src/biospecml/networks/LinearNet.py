import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, in_features, out_features, dropout_rate=None, residual_mode=False):
        super(LinearNetBasicBlock, self).__init__()
        self.residual_mode = residual_mode
        # self.transform_identity = nn.Linear(in_features, out_features)
        # layers in this block
        layers = [
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features),
            nn.ReLU(),
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
        ]
        self.activation_layer = nn.ReLU()
        if dropout_rate != None:
            self.dropout_layer = nn.Dropout(p=dropout_rate)
        else:
            self.dropout_layer = None

        # unpack layers
        self.block = nn.Sequential(*layers)

    def __str__(self):
        layers = [f'{self.block}']
        layers.append(f'{self.activation_layer}')
        if self.dropout_layer is not None:
            layers.append(f'{self.dropout_layer}')
        return '\n'.join(layers)

    def forward(self, x):
        if self.residual_mode:
            identity = x
            x = self.block(x)
            padding_size =  x.shape[1] - identity.shape[1]
            identity = nn.functional.pad(identity, (0, padding_size))
            x = identity + x
            x = self.activation_layer(x)
        else:
            x = self.block(x)
            x = self.activation_layer(x)
        if self.dropout_layer is not None:
            self.dropout_layer
        return x


class LinearNet(nn.Module):
    """
    A Linear NN that allows multiple blocks.

    Args:
    - input_size (int): Input node for the first layer.
    - hidden_size (int): Size of the hidden layers.
    - num_classes (int): Number of output classes.
    - hidden_expansion (str, optional): Option to expand hidden layer size for each block.
                                         Choose from 'double', 'same', 'half'. Default is 'same'.
    - dropout_rate (float, optional): Dropout rate for dropout layers. Default is None.
    - residual_mode (bool, optional): If True, enables residual connections between blocks. Default is False.
    - add_num_blocks (int, optional): Number of additional blocks to add. Default is 0.
    - use_softmax (bool, optional): If True, applies softmax activation to the output. Default is True.
    """
    def __init__(self, input_size:int, hidden_size:int, num_classes:int,
                 hidden_expansion:str='same', dropout_rate:float=None,
                 residual_mode:bool=False, add_num_blocks:int=1, use_softmax:bool=True):
        super(LinearNet, self).__init__()
        self.residual_mode = residual_mode
        self.use_softmax = use_softmax
        self.first_layer = nn.Linear(input_size, hidden_size)
        self.hidden_expansion = hidden_expansion
        self.basic_blocks = nn.ModuleList()

        if hidden_expansion == 'double' or hidden_expansion == 'same':
            current_size = hidden_size
        elif hidden_expansion == 'half':
            current_size = hidden_size

        for _ in range(add_num_blocks):
            if hidden_expansion == 'double':
                current_size *= 2
            elif hidden_expansion == 'half':
                current_size //= 2
            self.basic_blocks.append(
                # LinearNetBasicBlock(current_size // 2, current_size, dropout_rate, residual_mode)
                LinearNetBasicBlock(hidden_size, current_size, dropout_rate, residual_mode)
            )
            hidden_size = current_size

        self.classification_layer = nn.Linear(current_size, num_classes)

    def __str__(self):
        layers = [
            f'First layer: {self.first_layer}',
        ]
        if self.residual_mode:
            layers.append('Residual mode is ON for each following block(s).')
        layers += [f'Additional Block {i}: {block}' for i, block in enumerate(self.basic_blocks, 1)]
        layers.append(f'Classification Layer: {self.classification_layer}')
        if self.use_softmax:
            layers.append(f'Softmax Layer.')
        return '\n'.join(layers)

    def forward(self, x):

        x = self.first_layer(x)

        # check and add additional blocks
        if len(self.basic_blocks) > 0:
            for basic_block in self.basic_blocks:
                x = basic_block(x)

        x = self.classification_layer(x)

        if self.use_softmax:
            x = F.softmax(x, dim=1)
        return x


# class LinearNetBasicBlock(nn.Module):
#     """ THIS NETWORK IS PREVIOUSLY USED IN THE PDX N BLOCKS OPTIMISATIONS, BUT HAS INCCORECT IMPLEMENTATION OF RESISUAL CONNECTION
#     A Basic block for the linear net inspired by the ResNet block.

#     Args:
#     - in_features (int): Number of input features.
#     - out_features (int): Number of output features.
#     - dropout_rate (float, optional): Dropout rate for dropout layers. Default is None.
#     """
#     def __init__(self, in_features, out_features, dropout_rate=None, residual_mode=False):
#         super(LinearNetBasicBlock, self).__init__()
#         self.residual_mode = residual_mode
#         # layers in this block
#         layers = [
#             nn.Linear(in_features, out_features),
#             nn.BatchNorm1d(out_features),
#             nn.ReLU(),
#             nn.Linear(out_features, out_features),
#             nn.BatchNorm1d(out_features),
#         ]
#         self.activation_layer = nn.ReLU()
#         if dropout_rate != None: 
#             self.dropout_layer = nn.Dropout(p=dropout_rate)
#         else:
#             self.dropout_layer = None

#         # unpack layers
#         self.block = nn.Sequential(*layers)
    
#     def __str__(self):
#         layers = [f'{self.block}']
#         layers.append(f'{self.activation_layer}')
#         if self.dropout_layer is not None:
#             layers.append(f'{self.dropout_layer}')
#         return '\n'.join(layers)

#     def forward(self, x):
#         if self.residual_mode:
#             identity = x
#             x = self.block(x)
#             x = identity + x
#             x = self.activation_layer(x)
#         else:
#             x = self.block(x)
#             x = self.activation_layer(x)
#         if self.dropout_layer is not None:
#             self.dropout_layer
#         return x


# class LinearNet(nn.Module):
#     """
#     A Linear NN that allows multiple blocks.

#     Args:
#     - input_size (int): Input node for the first layer.
#     - hidden_size (int): Size of the hidden layers.
#     - num_classes (int): Number of output classes.
#     - dropout_rate (float, optional): Dropout rate for dropout layers. Default is None.
#     - residual_mode (bool, optional): If True, enables residual connections between blocks. Default is False.
#     - add_num_blocks (int, optional): Number of additional blocks to add. Default is 0.
#     - use_softmax (bool, optional): If True, applies softmax activation to the output. Default is True.
#     """
#     def __init__(self, input_size:int, hidden_size:int, num_classes:int,
#                  dropout_rate:float=None, residual_mode:bool=False,
#                  add_num_blocks:int=1, use_softmax:bool=True):
#         super(LinearNet, self).__init__()
#         self.residual_mode = residual_mode
#         self.use_softmax = use_softmax
#         self.first_layer = nn.Linear(input_size, hidden_size)  
#         self.basic_blocks = nn.ModuleList(
#             [LinearNetBasicBlock(hidden_size, hidden_size, dropout_rate, residual_mode) for _ in range(add_num_blocks)]
#             )
#         self.add_num_blocks = add_num_blocks
#         self.classification_layer = nn.Linear(hidden_size, num_classes)

#     def __str__(self):
#         layers = [
#             f'First layer: {self.first_layer}',
#         ]
#         if self.residual_mode:
#             layers.append('Residual mode is ON for each following block(s).')
#         layers += [f'Additional Block {i}: {block}' for i, block in enumerate(self.basic_blocks, 1)]
#         layers.append(f'Classification Layer: {self.classification_layer}')
#         if self.use_softmax:
#             layers.append(f'Softmax Layer.')
#         return '\n'.join(layers)

#     def forward(self, x):

#         x = self.first_layer(x)

#         # check and add additional blocks
#         if self.add_num_blocks > 0:
#             for basic_block in self.basic_blocks:
#                 x = basic_block(x)

#         x = self.classification_layer(x)

#         if self.use_softmax:
#             x = F.softmax(x, dim=1) # using dim i; (batch, i)
#         return x


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class LinearNetBasicBlock(nn.Module):
#     """
#     A Basic block for the linear net inspired by the ResNet block.

#     Args:
#     - in_features (int): Number of input features.
#     - out_features (int): Number of output features.
#     - dropout_rate (float, optional): Dropout rate for dropout layers. Default is None.
#     """
#     def __init__(self, in_features, out_features, dropout_rate=None):
#         super(LinearNetBasicBlock, self).__init__()
        
#         # layers in this block
#         layers = [
#             nn.Linear(in_features, out_features),
#             nn.BatchNorm1d(out_features),
#             nn.ReLU(),
#         ]
        
#         # add dropout layer if needed
#         if dropout_rate != None: 
#             layers.append(nn.Dropout(p=dropout_rate))
        
#         # unpack layers
#         self.block = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.block(x)


# class LinearNet(nn.Module):
#     """
#     A Linear NN that allows multiple blocks.

#     Args:
#     - input_size (int): Input node for the first layer.
#     - hidden_size (int): Size of the hidden layers.
#     - num_classes (int): Number of output classes.
#     - dropout_rate (float, optional): Dropout rate for dropout layers. Default is None.
#     - residual_mode (bool, optional): If True, enables residual connections between blocks. Default is False.
#     - add_num_blocks (int, optional): Number of additional blocks to add. Default is 0.
#     - use_softmax (bool, optional): If True, applies softmax activation to the output. Default is True.
#     """
#     def __init__(self, input_size:int, hidden_size:int, num_classes:int,
#                  dropout_rate:float=None, residual_mode:bool=False,
#                  add_num_blocks:int=0, use_softmax:bool=True):
#         super(LinearNet, self).__init__()
#         self.residual_mode = residual_mode
#         self.use_softmax = use_softmax
#         self.first_block = LinearNetBasicBlock(input_size, hidden_size, dropout_rate)
#         self.match_dim_layer = nn.Linear(input_size, hidden_size)  
#         self.basic_blocks = nn.ModuleList(
#             [LinearNetBasicBlock(hidden_size, hidden_size, dropout_rate) for _ in range(add_num_blocks)]
#             )
#         self.add_num_blocks = add_num_blocks
#         self.classification_layer = nn.Linear(hidden_size, num_classes)
    
#     def __str__(self):
#         layers = [
#             f'First Block: {self.first_block}',
#         ]
#         layers += [f'Additional Block {i}: {block}' for i, block in enumerate(self.basic_blocks, 1)]
#         if self.use_softmax:
#             layers.append(f'Softmax Layer.')
#         else:    
#             layers.append(f'Classification Layer: {self.classification_layer}')
#         return '\n'.join(layers)
    
#     def forward(self, x):
        
#         # check and apply residual connection in first block
#         if self.residual_mode:
#             # apply match layer so that input dim = first_block output dim
#             # identity = self.match_dim_layer(x) # error: this layer introduce untrackable backwards layer
#             identity = x
#             x = self.first_block(x)
#             x += identity
#         else:
#             x = self.first_block(x)

#         # check and add additional blocks
#         if self.add_num_blocks > 0:
#             for block in self.basic_blocks:
#                 if self.residual_mode:
#                     identity = x
#                     x = block(x)
#                     x += identity
#                 else:
#                     x = block(x)

#         # final layer and/or add softmax layer
#         if self.use_softmax:
#             x = F.softmax(x, dim=1) # using dim i; (batch, i)
#         else:
#             x = self.classification_layer(x)
#         return x