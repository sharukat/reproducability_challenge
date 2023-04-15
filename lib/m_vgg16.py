import torch
import math
import torch.nn as nn
from torchvision import models

def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
        module.weight.data.normal_(0, math.sqrt(2. / n))
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.weight.data.normal_(0, 0.01)
        module.bias.data.zero_()


class VGG16(nn.Module):
    """
    ==========================================================================
    A PyTorch implementation of VGG16 architecture for image classification.
    ==========================================================================

    Args:
    num_classes (int): The number of output classes.

    Attributes:
    features (nn.Sequential): The convolutional layers of the VGG16 architecture.
    classifier (nn.Sequential): The fully connected layers of the VGG16 architecture.
    num_classes (int): The number of output classes.

    Methods:
    forward(x): The forward pass of the VGG16 architecture.
    """

    def __init__(self, num_classes):
      super(VGG16, self).__init__()
      self.num_classes = num_classes

      self.features = nn.Sequential(
          nn.Conv2d(3, 64, kernel_size=3, padding=1),
          nn.BatchNorm2d(64),
          nn.ReLU(inplace=True),

          nn.Conv2d(64, 64, kernel_size=3, padding=1),
          nn.BatchNorm2d(64),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2, stride=2),

          nn.Conv2d(64, 128, kernel_size=3, padding=1),
          nn.BatchNorm2d(128),
          nn.ReLU(inplace=True),

          nn.Conv2d(128, 128, kernel_size=3, padding=1),
          nn.BatchNorm2d(128),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2, stride=2),

          nn.Conv2d(128, 256, kernel_size=3, padding=1),
          nn.BatchNorm2d(256),
          nn.ReLU(inplace=True),

          nn.Conv2d(256, 256, kernel_size=3, padding=1),
          nn.BatchNorm2d(256),
          nn.ReLU(inplace=True),

          nn.Conv2d(256, 256, kernel_size=3, padding=1),
          nn.BatchNorm2d(256),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2, stride=2),

          nn.Conv2d(256, 512, kernel_size=3, padding=1),
          nn.BatchNorm2d(512),
          nn.ReLU(inplace=True),

          nn.Conv2d(512, 512, kernel_size=3, padding=1),
          nn.BatchNorm2d(512),
          nn.ReLU(inplace=True),

          nn.Conv2d(512, 512, kernel_size=3, padding=1),
          nn.BatchNorm2d(512),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2, stride=2),

          nn.Conv2d(512, 512, kernel_size=3, padding=1),
          nn.BatchNorm2d(512),
          nn.ReLU(inplace=True),

          nn.Conv2d(512, 512, kernel_size=3, padding=1),
          nn.BatchNorm2d(512),
          nn.ReLU(inplace=True),
          
          nn.Conv2d(512, 512, kernel_size=3, padding=1),
          nn.BatchNorm2d(512),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2, stride=2),
      )
      
      self.classifier = nn.Sequential(
          nn.Dropout(),
          nn.Linear(512, 512), 
          nn.ReLU(True), 
          nn.Dropout(), 
          nn.Linear(512 , 512), 
          nn.ReLU(True), 
          nn.Dropout(), 
          nn.Linear(512 , self.num_classes)
      )

      # Initialize weights
      self.apply(initialize_weights)


    def forward(self,x):
      x = self.features(x)
      x = x.view(x.size()[0], -1)
      x = self.classifier(x)
      return x




        