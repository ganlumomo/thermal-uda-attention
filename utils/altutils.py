import os
import torch
import torchvision
from torchvision import datasets, transforms
import torch.utils.data as data
from torch.autograd import Variable
import configparser
import logging
from utils.imageFileList import ImageFileList
from utils.imageFileListWeightDomain import ImageFileListWeightDomain


def readConfigFile(filePath):
    """
    Read config file

    Args:
        filePath ([str]): path to config file

    Returns:
        [Obj]: config object
    """    
    config = configparser.ConfigParser()
    config.read(filePath)
    return config


def setLogger(logFilePath):
    """
    Set logger

    Args:
        logFilePath ([str]): path to log file

    Returns:
        [obj]: logger object
    """    
    logHandler = [logging.FileHandler(logFilePath), logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=logHandler)
    logger = logging.getLogger()
    return logger


def make_variable(tensor, device, volatile=False):
    """Convert Tensor to Variable."""
    if torch.cuda.is_available():
        tensor = tensor.to(device)
    return Variable(tensor, volatile=volatile)


def make_weight_for_balanced_classes(images, nclasses):
    count = [0]*nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.]*nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0]*len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight


def get_mscoco(dataset_root, batch_size, train):
    """Get MSCOCO datasets loader

    Args:
        dataset_root (str): path to the dataset folder
        batch_size (int): batch size
        train (bool): create loader for training or test set

    Returns:
        obj: dataloader object for MSCOCO dataset
    """  
    if train:
        pre_process = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4017, 0.3791, 0.3656), std=(0.2093, 0.2019, 0.1996))
        ])
        mscoco_dataset = datasets.ImageFolder(root=os.path.join(dataset_root, 'uda_data/mscoco2/train'),
                                             transform=pre_process)

        weight = make_weight_for_balanced_classes(mscoco_dataset.imgs, len(mscoco_dataset.classes))
        weight=torch.DoubleTensor(weight)

        sampler = torch.utils.data.sampler.WeightedRandomSampler(weight, len(weight))
        mscoco_data_loader = torch.utils.data.DataLoader(
            dataset=mscoco_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler, num_workers=4, pin_memory=True, drop_last=True)
    else:
        pre_process = transforms.Compose([transforms.Resize((224, 224)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(
                                              mean=(0.3961, 0.3743, 0.3603),
                                              std=(0.2086, 0.2012, 0.1987))])
        mscoco_dataset = datasets.ImageFolder(root=os.path.join(dataset_root, 'uda_data/mscoco2/val'),
                                            transform=pre_process)

        mscoco_data_loader = torch.utils.data.DataLoader(
            dataset=mscoco_dataset,
            batch_size=batch_size,
            shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    return mscoco_data_loader


def get_m3fd(dataset_root, batch_size, train):
    """Get M3FD datasets loader

    Args:
        dataset_root (str): path to the dataset folder
        batch_size (int): batch size
        train (bool): create loader for training or test set

    Returns:
        obj: dataloader object for M3FD dataset
    """ 
    # dataset and data loader
    if train:
        pre_process = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=(0.4821, 0.4821, 0.4821),
                                          std=(0.2081, 0.2081, 0.2081))])
        m3fd_dataset = datasets.ImageFolder(root=os.path.join(dataset_root, 'uda_data/m3fd/train'),
                                             transform=pre_process)
        weight = make_weight_for_balanced_classes(m3fd_dataset.imgs, len(m3fd_dataset.classes))
        weight=torch.DoubleTensor(weight)

        sampler = torch.utils.data.sampler.WeightedRandomSampler(weight, len(weight))

        m3fd_data_loader = torch.utils.data.DataLoader(
            dataset=m3fd_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler, num_workers=4, pin_memory=True, drop_last=True)
    else:
        pre_process = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=(0.4810, 0.4810, 0.4810),
                                          std=(0.2081, 0.2081, 0.2081))])
        m3fd_dataset = datasets.ImageFolder(root=os.path.join(dataset_root, 'uda_data/m3fd/val'),
                                            transform=pre_process)
        m3fd_data_loader = torch.utils.data.DataLoader(
            dataset=m3fd_dataset,
            batch_size=batch_size,
            shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    return m3fd_data_loader


def get_flir(dataset_root, batch_size, train, pseudo_label=False):
    """Get FLIR datasets loader

    Args:
        dataset_root (str): path to the dataset folder
        batch_size (int): batch size
        train (bool): create loader for training or test set

    Returns:
        obj: dataloader object for FLIR dataset
    """ 
    # data loader for pseudo label
    if pseudo_label:
        pre_process = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=(0.5776, 0.5776, 0.5776),
                                          std=(0.1319, 0.1319, 0.1319))])
        flir_dataset = datasets.ImageFolder(root=os.path.join(dataset_root, 'uda_data/flir/train'),
                                             transform=pre_process)

        flir_data_loader = torch.utils.data.DataLoader(
            dataset=flir_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1, pin_memory=True, drop_last=False)

        return flir_data_loader, flir_dataset.samples

    # dataset and data loader
    if train:
        pre_process = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=(0.5776, 0.5776, 0.5776),
                                          std=(0.1319, 0.1319, 0.1319))])
        flir_dataset = datasets.ImageFolder(root=os.path.join(dataset_root, 'uda_data/flir/train'),
                                             transform=pre_process)
        weight = make_weight_for_balanced_classes(flir_dataset.imgs, len(flir_dataset.classes))
        weight=torch.DoubleTensor(weight)

        sampler = torch.utils.data.sampler.WeightedRandomSampler(weight, len(weight))

        flir_data_loader = torch.utils.data.DataLoader(
            dataset=flir_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler, num_workers=4, pin_memory=True, drop_last=True)
    else:
        pre_process = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=(0.5587, 0.5587, 0.558),
                                          std=(0.1394, 0.1394, 0.1394))])
        flir_dataset = datasets.ImageFolder(root=os.path.join(dataset_root, 'uda_data/flir/val'),
                                            transform=pre_process)
        flir_data_loader = torch.utils.data.DataLoader(
            dataset=flir_dataset,
            batch_size=batch_size,
            shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    return flir_data_loader


def get_flir_from_list_wdomain(dataset_root, batch_size, train):
    """Get FLIR datasets loader

    Args:
        dataset_root (str): path to the dataset folder
        batch_size (int): batch size
        train (bool): create loader for training or test set

    Returns:
        obj: dataloader object for FLIR dataset
    """ 
    # dataset and data loader
    if train:
        pre_process = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=(0.5776, 0.5776, 0.5776),
                                          std=(0.1319, 0.1319, 0.1319))])
        flir_dataset = ImageFileListWeightDomain(root=dataset_root, imageFolder='uda_data/flir/train', flist=os.path.join(dataset_root, 'uda_data/flir', 'pseudo_labels_flir.txt'),
                                            transform=pre_process)
        weight = flir_dataset.weights
        weight = torch.DoubleTensor(weight)

        sampler = torch.utils.data.sampler.WeightedRandomSampler(weight, len(weight))

        flir_data_loader = torch.utils.data.DataLoader(
            dataset=flir_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler, num_workers=4, pin_memory=True, drop_last=True)
    else:
        pre_process = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=(0.5587, 0.5587, 0.558),
                                          std=(0.1394, 0.1394, 0.1394))])
        flir_dataset = ImageFileList(root=dataset_root, imageFolder='uda_data/flir/val', flist=os.path.join(dataset_root, 'uda_data/flir', 'test_wconf_wdomain_weights.txt'),
                                            transform=pre_process)
        flir_data_loader = torch.utils.data.DataLoader(
            dataset=flir_dataset,
            batch_size=batch_size,
            shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    return flir_data_loader
