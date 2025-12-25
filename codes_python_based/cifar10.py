from torch.utils.data import random_split, DataLoader
from torchvision.datasets import CIFAR10

def get_cifar10_loaders(preprocess, args):
    train_full = CIFAR10(
        root=args.data_root,
        train=True,
        download=True,
        transform=preprocess
    )
    train_set, val_set = random_split(
        train_full, [args.train_size, args.val_size]
    )
    test_set = CIFAR10(
        root=args.data_root,
        train=False,
        download=True,
        transform=preprocess
    )
    train_loader = DataLoader(
        train_set, batch_size=args.train_batch_size,
        shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_set, batch_size=args.val_batch_size,
        shuffle=False, num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=args.test_batch_size,
        shuffle=False, num_workers=args.num_workers
    )
    return train_loader, val_loader, test_loader, test_set.classes
