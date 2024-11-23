import pandas as pd

from torch.utils.data import DataLoader, ConcatDataset
from datasets import load_dataset
from wilds import get_dataset
from torchvision.datasets.folder import pil_loader
from torchvision.datasets import \
    ImageFolder, \
    CIFAR100, \
    CIFAR10, \
    Food101, \
    INaturalist, \
    DTD, \
    OxfordIIITPet, \
    PCAM

from utils import *


def train_val_dataloaders(args):
    if args.dataset is None:
        return None, None, None

    train_datasets = []
    val_datasets = []
    names = []
    sampling_ratios = []

    train_transform = make_transforms(**args.train_transform)
    val_transform = make_transforms(**args.val_transform)

    def append_and_print(train_ds, val_ds, short_name, long_name, ratio, subset=None):
        if subset is not None:
            train_ds, _ = split_dataset(dataset=train_ds, split=subset, seed=args.seed)
        train_datasets.append(train_ds)
        val_datasets.append(val_ds)
        names.append(short_name)
        sampling_ratios.append(ratio)
        print_ds_summary(name=long_name + ', ratio = {}'.format(ratio), num_train=len(train_ds), num_val=len(val_ds))

    if args.dataset_imagenet1k_use:
        train_dataset = ImageNetDataset(**args.dataset_imagenet1k_train, transform=train_transform)
        if args.dataset_imagenet1k_label_subset is not None:
            train_dataset = utils.imagenet_subset_samples(train_dataset, args.dataset_imagenet1k_label_subset)
        val_dataset = ImageNetDataset(**args.dataset_imagenet1k_val, transform=val_transform)

        append_and_print(train_dataset, val_dataset, 'in1k', 'ImageNet-1K',
                         args.dataset_imagenet1k_ratio, args.dataset_imagenet1k_subset)

    if args.dataset_imagenet21k_use:
        train_dataset = ImageNetDataset(**args.dataset_imagenet21k, transform=train_transform)
        val_dataset = ImageNetDataset(**args.dataset_imagenet1k_val, transform=val_transform)  # use in1k for validation

        append_and_print(train_dataset, val_dataset, 'in21k', 'ImageNet-21K',
                         args.dataset_imagenet21k_ratio, args.dataset_imagenet21k_subset)

    if args.dataset_cifar100_use:
        train_dataset = CIFAR100Dataset(**args.dataset_cifar100, train=True, transform=train_transform)
        val_dataset = CIFAR100Dataset(**args.dataset_cifar100, train=False, transform=val_transform)

        append_and_print(train_dataset, val_dataset, 'cifar100', 'CIFAR100',
                         args.dataset_cifar100_ratio, args.dataset_cifar100_subset)

    if args.dataset_cifar10_use:
        train_dataset = CIFAR10Dataset(**args.dataset_cifar10, train=True, transform=train_transform)
        val_dataset = CIFAR10Dataset(**args.dataset_cifar10, train=False, transform=val_transform)

        append_and_print(train_dataset, val_dataset, 'cifar10', 'CIFAR10',
                         args.dataset_cifar10_ratio, args.dataset_cifar10_subset)
        
    if args.dataset_food101_use:
        train_dataset = Food101Dataset(**args.dataset_food101, split='train', transform=train_transform)
        val_dataset = Food101Dataset(**args.dataset_food101, split='test', transform=val_transform)

        append_and_print(train_dataset, val_dataset, 'food101', 'Food101',
                         args.dataset_food101_ratio, args.dataset_food101_subset)

    if args.dataset_inat18_use:
        train_dataset = INat18Dataset(**args.dataset_inat18_train, transform=train_transform)
        val_dataset = INat18Dataset(**args.dataset_inat18_val, transform=val_transform)

        append_and_print(train_dataset, val_dataset, 'inat18', 'iNaturalist18',
                         args.dataset_inat18_ratio, args.dataset_inat18_subset)

    if args.dataset_dtd_use:
        train_dataset = ConcatDataset([DTDDataset(**args.dataset_dtd, split='train', transform=train_transform),
                                       DTDDataset(**args.dataset_dtd, split='val', transform=train_transform)])
        val_dataset = DTDDataset(**args.dataset_dtd, split='test', transform=val_transform)

        append_and_print(train_dataset, val_dataset, 'dtd', 'DTD',
                         args.dataset_dtd_ratio, args.dataset_dtd_subset)
        
    if args.dataset_cars_use:
        train_dataset = CarsDataset(**args.dataset_cars, split='train', transform=train_transform)
        val_dataset = CarsDataset(**args.dataset_cars, split='test', transform=val_transform)

        append_and_print(train_dataset, val_dataset, 'cars', 'StanfordCars',
                         args.dataset_cars_ratio, args.dataset_cars_subset)

    if args.dataset_pets_use:
        train_dataset = PetsDataset(**args.dataset_pets, split='trainval', transform=train_transform)
        val_dataset = PetsDataset(**args.dataset_pets, split='test', transform=val_transform)

        append_and_print(train_dataset, val_dataset, 'pets', 'OxfordIIITPet',
                         args.dataset_pets_ratio, args.dataset_pets_subset)
        
    if args.dataset_eurosat_use:
        train_dataset = EuroSATDataset(**args.dataset_eurosat, split='train', transform=train_transform)
        val_dataset = EuroSATDataset(**args.dataset_eurosat, split='test', transform=val_transform)

        append_and_print(train_dataset, val_dataset, 'eurosat', 'EuroSAT',
                         args.dataset_eurosat_ratio, args.dataset_eurosat_subset)
    
    if args.dataset_pcam_use:
        train_dataset = PCAMDataset(**args.dataset_pcam, split='train', transform=train_transform)
        val_dataset = PCAMDataset(**args.dataset_pcam, split='val', transform=val_transform)

        append_and_print(train_dataset, val_dataset, 'pcam', 'PCAM',
                         args.dataset_pcam_ratio, args.dataset_pcam_subset)

    if args.dataset_camelyon17_use:
        train_dataset = Camelyon17Dataset(**args.dataset_camelyon17, split='id_train', transform=train_transform)
        val_dataset = Camelyon17Dataset(**args.dataset_camelyon17, split='ood_val', transform=val_transform)

        append_and_print(train_dataset, val_dataset, 'camelyon17', 'Camelyon17',
                         args.dataset_camelyon17_ratio, args.dataset_camelyon17_subset)

    if args.dataset_fmow_use:
        train_dataset = FMoWDataset(**args.dataset_fmow, split='train', transform=train_transform)
        val_dataset = FMoWDataset(**args.dataset_fmow, split='val', transform=val_transform)

        append_and_print(train_dataset, val_dataset, 'fmow', 'fMoW',
                         args.dataset_fmow_ratio, args.dataset_fmow_subset)
        
    if args.dataset_ig_use:
        train_dataset = InfographDataset(**args.dataset_ig_train, root=args.dataset_ig_root, transform=train_transform)
        val_dataset = InfographDataset(**args.dataset_ig_val, root=args.dataset_ig_root, transform=val_transform)

        append_and_print(train_dataset, val_dataset, 'ig', 'Infograph',
                         args.dataset_ig_ratio, args.dataset_ig_subset)
    
    if args.dataset_iwildcam_use:
        train_dataset = IWildCamDataset(**args.dataset_iwildcam, split='train', transform=train_transform)
        val_dataset = IWildCamDataset(**args.dataset_iwildcam, split='val', transform=val_transform)

        append_and_print(train_dataset, val_dataset, 'iwildcam', 'iWildCam',
                         args.dataset_iwildcam_ratio, args.dataset_iwildcam_subset)
    
    if args.dataset_rxrx1_use:
        train_dataset = RXRX1Dataset(**args.dataset_rxrx1, split='train', transform=train_transform)
        val_dataset = RXRX1Dataset(**args.dataset_rxrx1, split='val', transform=val_transform)

        append_and_print(train_dataset, val_dataset, 'rxrx1', 'RXRX1',
                         args.dataset_rxrx1_ratio, args.dataset_rxrx1_subset)
        
    sampler = None
    train_dataset = ConcatDataset(train_datasets)

    if args.dataset_sampler == 'ratio':
        sampler = sampler_with_ratio(train_datasets, sampling_ratios, args)
        print('*' * 30, 'Ratio Sampler', '*' * 30)
        print('ratios sum up to {}'.format(sum(sampling_ratios)))
        print('*' * 60)
    elif args.dataset_sampler is not None:
        raise Exception('invalid sampler chosen - {}'.format(args.dataset_sampler))

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=sampler is None, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=args.train_pin_memory, drop_last=True)

    val_loaders = [DataLoader(val_ds, batch_size=args.val_batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=args.val_pin_memory)
                   for val_ds in val_datasets]

    print_ds_summary(name='Total', num_train=len(train_dataset), num_val=sum([len(ds) for ds in val_datasets]))

    return train_loader, val_loaders, names


class ImageNetDataset(ImageFolder):
    def __init__(self, root, transform=None, **kwargs):
        super(ImageNetDataset, self).__init__(root=root, transform=transform)

    @safe_getitem
    def __getitem__(self, idx):
        image, target = super(ImageNetDataset, self).__getitem__(idx)
        return image, target


class CIFAR100Dataset(CIFAR100):
    def __init__(self, root, train=True, transform=None, download=True, **kwargs):
        super(CIFAR100Dataset, self).__init__(root=root, train=train, transform=transform, download=download)

    def __getitem__(self, idx):
        image, target = super(CIFAR100Dataset, self).__getitem__(idx)
        return image, target


class CIFAR10Dataset(CIFAR10):
    def __init__(self, root, train=True, transform=None, download=True, **kwargs):
        super(CIFAR10Dataset, self).__init__(root=root, train=train, transform=transform, download=download)

    def __getitem__(self, idx):
        image, target = super(CIFAR10Dataset, self).__getitem__(idx)
        return image, target


class Food101Dataset(Food101):
    def __init__(self, root, split='train', transform=None, download=True, **kwargs):
        super(Food101Dataset, self).__init__(root=root, split=split, transform=transform, download=download)

    def __getitem__(self, idx):
        image, target = super(Food101Dataset, self).__getitem__(idx)
        return image, target


class INat18Dataset(INaturalist):
    def __init__(self, root, version='2018', transform=None, download=False, **kwargs):
        super(INat18Dataset, self).__init__(root=root, version=version, download=download)
        self.transform_ = transform

    def __getitem__(self, idx):
        image, target = super(INat18Dataset, self).__getitem__(idx)
        image = image.convert('RGB')  # need to convert to RGB since some images are greyscale

        if self.transform_:
            image = self.transform_(image)

        return image, target


class DTDDataset(DTD):
    def __init__(self, root, split='train', transform=None, download=True, **kwargs):
        super(DTDDataset, self).__init__(root=root, split=split, transform=transform, download=download)

    def __getitem__(self, idx):
        image, target = super(DTDDataset, self).__getitem__(idx)
        return image, target


class CarsDataset(Dataset):
    def __init__(self, root, split='train', transform=None, **kwargs):
        self.data = load_dataset("tanganke/stanford_cars", cache_dir=root, split=split)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image, target = sample['image'], sample['label']
        image = image.convert('RGB')  # need to convert to RGB since some images are greyscale

        if self.transform:
            image = self.transform(image)

        return image, target


class PetsDataset(OxfordIIITPet):
    def __init__(self, root, split='trainval', transform=None, download=True, **kwargs):
        super(PetsDataset, self).__init__(root=root, split=split, transform=transform, download=download)

    def __getitem__(self, idx):
        image, target = super(PetsDataset, self).__getitem__(idx)
        return image, target


class EuroSATDataset(Dataset):
    def __init__(self, root, split='train', transform=None, **kwargs):
        self.data = load_dataset("tanganke/eurosat", cache_dir=root, split=split)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image, target = sample['image'], sample['label']

        if self.transform:
            image = self.transform(image)

        return image, target


class PCAMDataset(PCAM):
    def __init__(self, root, split='train', transform=None, download=True, **kwargs):
        super(PCAMDataset, self).__init__(root=root, split=split, transform=transform, download=download)

    def __getitem__(self, idx):
        image, target = super(PCAMDataset, self).__getitem__(idx)
        return image, target


class Camelyon17Dataset(Dataset):
    def __init__(self, root, split='id_train', transform=None, **kwargs):
        self.data = load_dataset("jxie/camelyon17", cache_dir=root, split=split)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image, target = sample['image'], sample['label']

        if self.transform:
            image = self.transform(image)

        return image, target


class FMoWDataset(Dataset):
    def __init__(self, root, split='train', transform=None, **kwargs):
        self.data = load_dataset("danielz01/fMoW", cache_dir=root, split=split)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image, target = sample['image'], sample['label']

        if self.transform:
            image = self.transform(image)

        return image, target


class InfographDataset(Dataset):
    def __init__(self, root, txt_path, transform=None, **kwargs):
        self.root = root
        self.transform = transform

        self.data = pd.read_csv(txt_path, sep=' ', header=None, names=['img_path', 'target'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, target = self.data['img_path'][idx], self.data['target'][idx]

        img_path = os.path.join(self.root, img_path)
        image = pil_loader(img_path)

        if self.transform:
            image = self.transform(image)

        return image, target


class IWildCamDataset(Dataset):
    def __init__(self, root, split='train', transform=None, download=True, **kwargs):
        self.data = get_dataset(dataset='iwildcam',
                                root_dir=root,
                                download=download).get_subset(split=split, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, target, _ = self.data[idx]
        return image, target


class RXRX1Dataset(Dataset):
    def __init__(self, root, split='train', transform=None, download=True, **kwargs):
        self.data = get_dataset(dataset='rxrx1',
                                root_dir=root,
                                download=download).get_subset(split=split, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, target, _ = self.data[idx]
        return image, target
