import torch.utils.data as data

from PIL import Image
import os
import os.path

def default_loader(path):
    return Image.open(path).convert('RGB')

def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    weights = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath, imlabel, confidence, discOut, discConf, weight = line.strip().split()
            weights.append(float(weight))
            imlist.append((impath, int(imlabel), float(confidence), int(discOut), float(discConf)))

    return imlist, weights

class ImageFileListWeightDomain(data.Dataset):
    def __init__(self, root, imageFolder, flist, transform=None, target_transform=None, 
    flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imageFolder = imageFolder
        self.imlist, self.weights = flist_reader(flist)		
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        impath, label, confidence, discOut, discConf = self.imlist[index]
        img = self.loader(os.path.join(self.root, self.imageFolder, impath))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        
        return img, label, confidence, discOut, discConf
        
    def __len__(self):
        return len(self.imlist)
