# three dense blocks that each has an equal number of layer
#  Before entering the first dense block, a convolution with 16 (or twice the growth rate for DenseNet-BC)
import torch as nn

class DenseBlock(nn.Module):
  def __init__(self):
    pass

class TransitionBlock(nn.Module):
  def __init__(self):
    pass

class DenseNet(nn.Module):
  def __init__(self, depth, num_class, growth_rate, reduction, drop_rate):
    super(DenseNet, self).__init__()
    db_layers = (depth - 4) / 3