from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

def loader(dataset_name, batch_size):
    path = '/content/drive/MyDrive/NN_Course_Project/project/datasets'
    allowed_datasets = ['CIFAR10', 'CIFAR100','SVNH']

    if dataset_name not in allowed_datasets:
        raise ValueError(f"Invalid dataset name {dataset_name}. Allowed datasets are: {allowed_datasets}")
    
    else:

        if dataset_name == 'CIFAR10':
            mean = [0.491, 0.482, 0.447]
            std = [0.247, 0.243, 0.262]
        else:
            mean = [0.507, 0.487, 0.441]
            std = [0.267, 0.256, 0.276]


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

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Length of train dataloader: {len(train_loader)} batches of {batch_size}")
    print(f"Length of test dataloader: {len(test_loader)} batches of {batch_size}")

    return train_loader, test_loader