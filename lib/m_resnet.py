import torch
import torch.nn as nn
    


class BaseBlock(nn.Module):
    """
    A basic building block for a neural network, consisting of two convolutional layers with batch normalization and ReLU activation functions.
    =========================================================================================================================================

    Args:
        in_channels (int): The number of input channels to the first convolutional layer.
        out_channels (int): The number of output channels from the second convolutional layer.
        stride (int, optional): The stride of the convolutional layers. Defaults to 1.
        downsample (torch.nn.Module, optional): An optional downsampling module to be applied to the input. Defaults to None.
    """


    def __init__(self, in_channels, out_channels, stride=1, downsample=None, expansion=4):
        """
        Initializes the BaseBlock module.
        =================================

        Args:
            in_channels (int): The number of input channels to the first convolutional layer.
            out_channels (int): The number of output channels from the second convolutional layer.
            stride (int, optional): The stride of the convolutional layers. Defaults to 1.
            downsample (torch.nn.Module, optional): An optional downsampling module to be applied to the input. Defaults to None.
        """
        
        self.EXPANSION = expansion
        super(BaseBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.downsample = downsample

        self.pre_activ_layers = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True)
            )

        self.layers = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=self.stride, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=self.stride, padding=1, bias=False),
            )


    def forward(self, x):
        """
        Performs a forward pass through the BaseBlock module.
        =====================================================

        Args:
            x (torch.Tensor): The input tensor to the module.

        Returns:
            torch.Tensor: The output tensor of the module.
        """


        residual = x
        output = self.pre_activ_layers(x)

        if self.downsample is not None:
            residual = self.downsample(x)

        output = self.layers(output)
        output += residual
        return output



class ResNET(nn.Module):
    def __init__(self, block, layers, output_classes):
        super(ResNET, self).__init__()

        self.IN_CHANNELS = 16
        self.block = block
        self.layers = layers
        self.output_classes = output_classes

        # Define the covolutional layer
        self.conv_l1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)

        # Define inner layers with residual blocks
        self.layer_1 = self._make_layer(block, 16, 16, layers[0])
        self.layer_2 = self._make_layer(block, 16, 32, layers[1], stride=2)
        self.layer_3 = self._make_layer(block, 32, 64, layers[2], stride=2)

        # Define the final layers
        self.classifier_block = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.AvgPool2d(8, stride=1),
            nn.Flatten(),
            nn.Linear(64, self.output_classes )
        )

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, block, out_channels, num_blocks, stride = 1, downsample=None):
        downsample = nn.Sequential(
            nn.Conv2d(in_channels=self.IN_CHANNELS, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)) if stride != 1 or self.IN_CHANNELS != out_channels else None # Ternary conditional operator
        
        layers = [block(self.IN_CHANNELS, out_channels, stride, downsample)]
        self.IN_CHANNELS = out_channels
        for k in range(1, num_blocks):
            layers.append(block(self.IN_CHANNELS, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        output = self.conv_l1(x)
        output = self.layer_1(output)
        output = self.layer_2(output)
        output = self.layer_3(output)
        output = self.classifier_block(output)
        return output

        