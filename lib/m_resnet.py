import torch
import torch.nn as nn
    


def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()


class BaseBlock(nn.Module):
    """
    =================================
    Initializes the BaseBlock module.
    =================================

    Args:
        in_channels (int): # input channels to the first convolutional layer.
        out_channels (int): # output channels from the second convolutional layer.
        stride (int, optional): The stride of the convolutional layers. Defaults to 1.
        downsample (torch.nn.Module, optional): An optional downsampling module to be applied to the input. Defaults to None.
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BaseBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample
        self.stride = stride

        self.pre_activ_layers = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True))

        self.layers = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            )

    def forward(self, x):
        residual = x
        output = self.pre_activ_layers(x)

        if self.downsample is not None:
            residual = self.downsample(output)

        output = self.layers(output)
        output += residual
        return output



class ResNET(nn.Module):
    """
    ==========================================================================
    A Residual Neural Network (ResNet) model.
    ==========================================================================

    Args:
        block (nn.Module): A residual block module.
        layers (list): A list of integers representing the number of residual blocks in each layer.
        num_classes (int): The number of classes in the dataset.

    Attributes:
        IN_CHANNELS (int): The number of input channels.
        conv_1 (nn.Conv2d): A convolutional layer with 3 input channels and 16 output channels.
        layer_block_1 (nn.Sequential): A sequence of residual blocks with 16 output channels.
        layer_block_2 (nn.Sequential): A sequence of residual blocks with 32 output channels.
        layer_block_3 (nn.Sequential): A sequence of residual blocks with 64 output channels.
        classifier_block (nn.Sequential): A sequence of layers that classify the input data.

    Methods:
        make_layer(block, out_channels, num_blocks, stride=1):
            Returns a sequence of residual blocks.

        forward(x):
            Passes input data through the network and returns the output.

    """

    def __init__(self, block, layers, num_classes):
        super(ResNET, self).__init__()
        self.IN_CHANNELS = 16
        self.num_classes = num_classes

        self.conv_1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer_block_1 = self.make_layer(block, 16, layers[0])
        self.layer_block_2 = self.make_layer(block, 32, layers[1], stride=2)
        self.layer_block_3 = self.make_layer(block, 64, layers[2], stride=2)

        # Define the final layers
        self.classifier_block = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(8, stride=1),
            nn.Flatten(),
            nn.Linear(64, self.num_classes)
        )

        # Initialize weights
        self.apply(initialize_weights)


    def make_layer(self, block, out_channels, num_blocks, stride = 1):
        """
        Returns a sequence of residual blocks.

        Args:
            block (nn.Module): A residual block module.
            out_channels (int): The number of output channels for each residual block.
            num_blocks (int): The number of residual blocks in the sequence.
            stride (int): The stride length for the first residual block in the sequence.

        Returns:
            nn.Sequential: A sequence of residual blocks.

        """
        downsample = nn.Sequential(
            nn.Conv2d(in_channels=self.IN_CHANNELS, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)) if stride != 1 or self.IN_CHANNELS != out_channels else None # Ternary conditional operator
      
        layers = [block(self.IN_CHANNELS, out_channels, stride, downsample)]
        self.IN_CHANNELS = out_channels
        for _ in range(1, num_blocks):
            layers.append(block(self.IN_CHANNELS, out_channels))
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv_1(x)
        x = self.layer_block_1(x)
        x = self.layer_block_2(x)
        x = self.layer_block_3(x)
        x = self.classifier_block(x)
        return x



def ResNet110(num_classes):
  model = ResNET(BaseBlock, [18,18,18], num_classes) 
  return model
        