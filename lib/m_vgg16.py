import torch
import torch.nn as nn
from torchvision import models

# class VGG16(nn.Module):
#     def __init__(self, num_classes: int):
#         super(VGG16, self).__init__()
#         self.num_classes = num_classes

#         # weights = models.VGG16_Weights.DEFAULT
#         model = models.vgg16(weights=None)

#         # for param in model.parameters():
#         #     param.requires_grad = False
        
#         n_inputs = model.classifier[6].in_features
#         model.classifier[6] = nn.Sequential(
#             nn.Dropout(),
#             nn.Linear(n_inputs,512),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(512,512),
#             nn.ReLU(True),
#             nn.Linear(512, self.num_classes)
#         )
        
#         self.model = model

#         for m in self.modules():
#           if isinstance(m, nn.Conv2d):
#               nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#               if m.bias is not None:
#                   nn.init.constant_(m.bias, 0)



#     def forward(self, x):
#         # x = self.features(x)
#         # x = x.view(x.size(0), -1)
#         x = self.model(x)
#         return x

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)


class VGG16(nn.Module):
    def __init__(self, num_classes):
      super(VGG16, self).__init__()
      self.num_classes = num_classes

      self.features = nn.Sequential(
          nn.Conv2d(3, 64, kernel_size=3, padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(64, 64, kernel_size=3, padding=1),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2, stride=2),
          nn.Conv2d(64, 128, kernel_size=3, padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(128, 128, kernel_size=3, padding=1),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2, stride=2),
          nn.Conv2d(128, 256, kernel_size=3, padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(256, 256, kernel_size=3, padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(256, 256, kernel_size=3, padding=1),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2, stride=2),
          nn.Conv2d(256, 512, kernel_size=3, padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(512, 512, kernel_size=3, padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(512, 512, kernel_size=3, padding=1),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2, stride=2),
          nn.Conv2d(512, 512, kernel_size=3, padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(512, 512, kernel_size=3, padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(512, 512, kernel_size=3, padding=1),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2,stride=2),
          nn.Flatten(),
          nn.Dropout(),
          nn.Linear(512 , 512), 
          nn.ReLU(True), 
          nn.Dropout(), 
          nn.Linear(512 , 512), 
          nn.ReLU(True), 
          nn.Dropout(), 
          nn.Linear(512 , self.num_classes)
      )
      
      # self.classifier = nn.Sequential(
      #     nn.Dropout(),
      #     nn.Linear(512 * 7 * 7 , 512), 
      #     nn.ReLU(True), 
      #     nn.Dropout(), 
      #     nn.Linear(512 , 512), 
      #     nn.ReLU(True), 
      #     nn.Dropout(), 
      #     nn.Linear(512 , self.num_classes)
      # )

      for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def forward(self,x):
      x = self.features(x)
      # x = x.view(x.size(0), -1)
      # x = self.classifier(x)
      return x




        