from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import lib.global_settings as settings

class BaseTransform:
  def __init__(self):
      pass

  def transform(self, mean, std):
      train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

      test_tf = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize(mean=mean, std=std)])

      return train_tf, test_tf


class CIFAR10Dataset(BaseTransform):
    def __init__(self):
        self.mean = [0.491, 0.482, 0.447]
        self.std  = [0.247, 0.243, 0.262]

        self.train_tf, self.test_tf = super().transform(self.mean, self.std)

    def load_data(self):
        train = datasets.CIFAR10(
            root=settings.DATA_PATH,
            train=True,
            download=True,
            transform=self.train_tf)
        
        test = datasets.CIFAR10(
            root=settings.DATA_PATH,
            train=False,
            download=False,
            transform=self.train_tf)

        return train, test


class CIFAR100Dataset(BaseTransform):
    def __init__(self):
        self.mean = [0.507, 0.487, 0.441]
        self.std  = [0.267, 0.256, 0.276]

        self.train_tf, self.test_tf = super().transform(self.mean, self.std)

    def load_data(self):
        train = datasets.CIFAR100(
            root=settings.DATA_PATH,
            train=True,
            download=True,
            transform=self.train_tf)
        
        test = datasets.CIFAR100(
            root=settings.DATA_PATH,
            train=False,
            download=False,
            transform=self.train_tf)

        return train, test


def loader(dataset_name):

    if dataset_name == 'CIFAR10':
        train_data, test_data = CIFAR10Dataset().load_data()
    elif dataset_name == 'CIFAR100':
        train_data, test_data = CIFAR100Dataset().load_data()
    else:
        raise ValueError(f"Invalid dataset name {dataset_name}. Allowed datasets are: {allowed_datasets}")

    train_loader = DataLoader(
      train_data, 
      batch_size=settings.BATCH_SIZE, 
      shuffle=True, 
      num_workers=2)

    test_loader = DataLoader(
      test_data, 
      batch_size=settings.BATCH_SIZE, 
      shuffle=False, 
      num_workers=2)

    print(f"Length of train dataloader: {len(train_loader)} batches of {settings.BATCH_SIZE}")
    print(f"Length of test dataloader: {len(test_loader)} batches of {settings.BATCH_SIZE}")

    return train_loader, test_loader