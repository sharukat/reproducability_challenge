import os
import torch

EPOCHS = 300
LR = 0.1
MOMENTUM = 0.9
CHECKPOINT_EPOCHS = [25, 50, 75, 100, 125, 150, 175, 200,225, 250, 275,300]
BATCH_SIZE = 128

# Cross Entropy Loss
CEL = torch.nn.CrossEntropyLoss()

# Marginal Ranking Loss
MRL = torch.nn.MarginRankingLoss(margin=0.0)

# Define paths
BASE_PATH = '/content/drive/MyDrive/NN_Course_Project/project'
MODEL_PATH  = os.path.join(BASE_PATH, 'models')
DATA_PATH   = os.path.join(BASE_PATH, 'datasets')