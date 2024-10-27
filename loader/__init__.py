import torchvision.datasets as datasets
import torchvision.transforms as transforms


def to_m1_1(x):
    """converts x to [-1, 1]"""
    return x * 2 - 1

def get_dataset(name, data_dir="datasets"):

    if name == "cifar10":
        transform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        train_set = datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=transform_train,
        )
    else:
        raise NotImplementedError

    return train_set 


