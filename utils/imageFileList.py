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
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath, imlabel, confidence = line.strip().split()
            imlist.append((impath, int(imlabel), float(confidence)))

    return imlist

class ImageFileList(data.Dataset):
    def __init__(self, root, imageFolder, flist, transform=None, target_transform=None, 
    flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imageFolder = imageFolder
        self.imlist = flist_reader(flist)		
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        impath, label, confidence = self.imlist[index]
        img = self.loader(os.path.join(self.root, self.imageFolder, impath))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        
        return img, label, confidence
        
    def __len__(self):
        return len(self.imlist)
