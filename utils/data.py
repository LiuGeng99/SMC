import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels
from . import autoaugment
from . import ops

class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None
    class_names = None


class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor(),
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
        ),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self, test_dir=None):
        train_dataset = datasets.cifar.CIFAR10("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )
        self.class_names = train_dataset.classes


class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor()
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
        ),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self, test_dir=None):
        train_dataset = datasets.cifar.CIFAR100("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )
        self.class_names = train_dataset.classes


class iCIFAR100_AA(iCIFAR100):
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
        autoaugment.CIFAR10Policy(),
        transforms.ToTensor(),
        ops.Cutout(n_holes=1, length=16),
    ]


class iCIFAR10_AA(iCIFAR10):
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
        autoaugment.CIFAR10Policy(),
        transforms.ToTensor(),
        ops.Cutout(n_holes=1, length=16),
    ]


class iImageNet1000(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self, test_dir_arg=None):
        train_dir = "/data3/geng_liu/ImageNet1K/train/"
        test_dir = test_dir_arg if test_dir_arg is not None else "/data3/geng_liu/ImageNet1K/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.class_names = train_dset.classes
        
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNet100(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        # transforms.TrivialAugmentWide(),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self, test_dir=None):
        train_dir = "/home/geng_liu/CL/dataset/ImageNet100/train/"
        mem_dir = "/home/geng_liu/CL/dataset/ImageNet100/compress/"
        if test_dir == None:
            test_dir = "/home/geng_liu/CL/dataset/ImageNet100/val/"

        train_dset = datasets.ImageFolder(train_dir)
        mem_dset = datasets.ImageFolder(mem_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.class_names = train_dset.classes

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.mem_data, self.mem_targets = split_images_labels(mem_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class iImageNet100_LT(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self, test_dir_arg=None):
        train_dir = "/data3/geng_liu/ImageNet-LT/train_min10/"
        mem_dir = "/data3/geng_liu/ImageNet-LT/compress/"
        test_dir = test_dir_arg if test_dir_arg is not None else "/data3/geng_liu/ImageNet100/val"

        train_dset = datasets.ImageFolder(train_dir)
        mem_dset = datasets.ImageFolder(mem_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.class_names = train_dset.classes
        
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.mem_data, self.mem_targets = split_images_labels(mem_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class iImageNet_OOD(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self, test_dir_arg=None):
        train_dir = "/home/geng_liu/CL/dataset/ImageNet100/train/"
        mem_dir = "/home/geng_liu/CL/dataset/ImageNet100/compress/"
        test_dir = test_dir_arg if test_dir_arg is not None else "/home/geng_liu/CL/dataset/imagenet-o"

        train_dset = datasets.ImageFolder(train_dir)
        mem_dset = datasets.ImageFolder(mem_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.class_names = train_dset.classes
        
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.mem_data, self.mem_targets = split_images_labels(mem_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
        
class iMed40(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(224),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(40).tolist()

    def download_data(self, test_dir=None):
        train_dir = "/home/geng_liu/CL/dataset/Med/data/train_500"
        mem_dir = "/home/geng_liu/CL/dataset/Med/compress/train_500"
        if test_dir is None:
            test_dir = "/home/geng_liu/CL/dataset/Med/data/val"

        train_dset = datasets.ImageFolder(train_dir)
        mem_dset = datasets.ImageFolder(mem_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.class_names = train_dset.classes
        
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.mem_data, self.mem_targets = split_images_labels(mem_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
        
class iCore50(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(),
    ]
    test_trsf = [
        transforms.Resize(224),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(50).tolist()

    def download_data(self, test_dir_arg=None):
        train_dir = "/home/geng_liu/CL/dataset/Core50/data/train_50"
        # mem_dir =   "/home/geng_liu/CL/dataset/Core50/data/train"
        mem_dir =   "/home/geng_liu/CL/dataset/Core50/data/memory_50" # 就是只保存100张的
        test_dir = test_dir_arg if test_dir_arg is not None else "/home/geng_liu/CL/dataset/Core50/data/val"

        train_dset = datasets.ImageFolder(train_dir)
        mem_dset = datasets.ImageFolder(mem_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.class_names = train_dset.classes
        
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.mem_data, self.mem_targets = split_images_labels(mem_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iLaryngo29(iData):
    use_path = False
    train_trsf = [
        transforms.Resize([224,224]),
        transforms.TrivialAugmentWide(),
        #transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        #transforms.RandomRotation(15),
        #transforms.RandomErasing(p=0.2), # 不确定随机擦除是否重要
        
        # transforms.RandomResizedCrop(224), # crop的程度可能需要改下 或者干脆不加
    ]
    test_trsf = [
        transforms.Resize([224,224]),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.568, 0.364, 0.329], std=[0.300, 0.213, 0.198]),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self, test_dir_arg=None):
        train_dir = "/data3/geng_liu/Hou_dataset/data_cropped/train"
        mem_dir = "/data3/geng_liu/Hou_dataset/data_cropped/compress_highres"
        test_dir = test_dir_arg if test_dir_arg is not None else "/data3/geng_liu/Hou_dataset/data_cropped/val"

        train_dset = datasets.ImageFolder(train_dir)
        mem_dset = datasets.ImageFolder(mem_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.class_names = train_dset.classes
        
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.mem_data, self.mem_targets = split_images_labels(mem_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)