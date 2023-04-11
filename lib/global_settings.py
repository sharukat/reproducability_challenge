import os
import torch

EPOCHS = 300
LR = 0.1
MOMENTUM = 0.9
CHECKPOINT_EPOCHS = [50, 100, 150, 200, 250, 300]
BATCH_SIZE = 128

# Cross Entropy Loss
CEL = torch.nn.CrossEntropyLoss()

# Marginal Ranking Loss
MRL = torch.nn.MarginRankingLoss(margin=0.0)

# Define paths
BASE_PATH = '/content/drive/MyDrive/NN_Course_Project/project'
MODEL_PATH  = os.path.join(BASE_PATH, 'models')
DATA_PATH   = os.path.join(BASE_PATH, 'datasets')