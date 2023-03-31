import torch

def negative_entropy(data):
    return -torch.sum(torch.log(torch.softmax(data, dim=1)) * torch.softmax(data, dim=1), dim=1)


def max_class_prob(data):
    output = torch.nn.Softmax(dim=1)(data)
    max_prob, max_class = torch.max(output, dim=1)
    return max_prob