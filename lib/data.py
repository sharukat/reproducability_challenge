from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import lib.global_settings as settings

def loader(dataset_name):
    path = '/content/drive/MyDrive/NN_Course_Project/project/datasets'
    allowed_datasets = ['CIFAR10', 'CIFAR100']

    if dataset_name not in allowed_datasets:
        raise ValueError(f"Invalid dataset name {dataset_name}. Allowed datasets are: {allowed_datasets}")
    
    else:

        if dataset_name == 'CIFAR10':
            mean = [0.491, 0.482, 0.447]
            std = [0.247, 0.243, 0.262]
        else:
            mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
            std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]


        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        train = getattr(datasets, dataset_name)(
            root=path,
            train=True,
            download=True,
            transform=train_tf, 
            target_transform=None)
        
        test = getattr(datasets, dataset_name)(
            root=path,
            train=False,
            download=True,
            transform=test_tf,
            target_transform=None)

    # Train and test data loaders
    train_loader = DataLoader(
      train, 
      batch_size=settings.BATCH_SIZE, 
      shuffle=True, 
      num_workers=2)

    test_loader = DataLoader(
      test, 
      batch_size=settings.BATCH_SIZE, 
      shuffle=False, 
      num_workers=2)

    print(f"Length of train dataloader: {len(train_loader)} batches of {settings.BATCH_SIZE}")
    print(f"Length of test dataloader: {len(test_loader)} batches of {settings.BATCH_SIZE}")

    return train_loader, test_loader