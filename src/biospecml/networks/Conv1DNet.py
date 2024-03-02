# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class Conv1DNetBasicBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, dropout_rate=0, padding=1):
#         super(Conv1DNetBasicBlock, self).__init__()

#         layers = [
#             nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
#             nn.BatchNorm1d(out_channels),
#             nn.ReLU(),
#         ]

#         if dropout_rate > 0:
#             layers.append(nn.Dropout(p=dropout_rate))

#         self.block = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.block(x)


# class Conv1DNet(nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0,
#                  residual_mode=False, add_num_blocks=0, use_softmax=True):
#         super(Conv1DNet, self).__init__()
#         self.residual_mode = residual_mode
#         self.use_softmax = use_softmax
#         self.first_block = Conv1DNetBasicBlock(1, hidden_size, 3, dropout_rate)  # Assuming input is 1 channel (e.g., time series)
#         self.basic_blocks = nn.ModuleList([
#             Conv1DNetBasicBlock(hidden_size, hidden_size, 3, dropout_rate) for _ in range(add_num_blocks)
#         ])
#         self.classification_layer = nn.Linear(hidden_size*input_size, num_classes)
#         self.add_num_blocks = add_num_blocks

#     def forward(self, x):
#         # x = x.unsqueeze(1)  # Add channel dimension for 1D convolution

#         if self.residual_mode:
#             x_resi = x
#             x = self.first_block(x) + x_resi
#         else:
#             x = self.first_block(x)

#         if self.add_num_blocks > 0:
#             for block in self.basic_blocks:
#                 if self.residual_mode:
#                     x_resi = x
#                     x = block(x) + x_resi
#                 else:
#                     x = block(x)

#         # x = x.squeeze(2)  # Remove the temporal dimension added by the convolution
#         x = x.view(x.size(0), -1) # flatten
#         x = self.classification_layer(x)
#         if self.use_softmax:
#             x = F.softmax(x, dim=1)
#             return x
#         else:
#             return x

# # --------------- set params for network ---------------

# input_size = 218        ## Replace with the actual input size of your data
# hidden_size = 512       ## Adjust as needed
# num_classes = 7         ## default = 7, Number of classes in your classification task
# dropout_rate = 0        ## dropout layer
# add_num_blocks = 0      ## Number of basic blocks (0,1,3,5,7,9)
# residual_mode = True    ## default = False, add residual connection in each layer

# model = Conv1DNet(input_size = input_size,
#                   hidden_size = hidden_size,
#                   num_classes = num_classes,
#                   dropout_rate = dropout_rate,
#                   add_num_blocks = add_num_blocks,
#                   residual_mode = residual_mode,
#                   use_softmax = False
#                   )
# print(model)