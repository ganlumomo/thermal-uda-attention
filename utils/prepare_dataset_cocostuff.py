import os
import glob
import cv2
import numpy as np
import tqdm
import csv
import shutil

cocostuff164k_images_train = '/home/carson/data/cocostuff/dataset/images/train2017'
cocostuff164k_images_val = '/home/carson/data/cocostuff/dataset/images/val2017'

cocostuff164k_annotations_train = '/home/carson/data/cocostuff/dataset/annotations/train2017'
cocostuff164k_annotations_val = '/home/carson/data/cocostuff/dataset/annotations/val2017'

def get_label_mapping():
    
    label_mapping = {}
    with open('./utils/cocostuff_labels.txt') as f:
        for line in f.readlines():
            idx, trainId, name = line.strip().split(':')
            label_mapping[int(idx)] = int(trainId)
    return label_mapping

def find_images_with_label_idx(idx_want, anns_path, imgs_path, annotations_out_dir, images_out_dir):
    annotations = sorted(glob.glob(os.path.join(anns_path, '*')))
    images = sorted(glob.glob(os.path.join(imgs_path, '*')))

    os.makedirs(annotations_out_dir, exist_ok=True)
    os.makedirs(images_out_dir, exist_ok=True)

    idx_want_set = set(idx_want)
    label_mapping = get_label_mapping()
    for annotation_path, img_path in tqdm.tqdm(zip(annotations, images), total=len(images)):
        assert os.path.basename(annotation_path).split('.')[0] == os.path.basename(img_path).split('.')[0]
        mask = cv2.imread(annotation_path, 0) + 1
        present_idx_set = set(np.unique(mask))

        common = present_idx_set.intersection(idx_want_set)
        if len(common) != 0:
            bin_mask = np.zeros_like(mask)
            for idx in idx_want_set:
                bin_mask = np.logical_or(mask==idx, bin_mask)
            if np.sum(bin_mask)/(mask.shape[0]*mask.shape[1]) > 0.2:
                new_mask = np.zeros_like(mask)
                for k, v in label_mapping.items():
                    new_mask[mask==k] = v
                new_mask = np.uint8(new_mask)
                print(np.unique(new_mask))
                new_mask_path = os.path.join(annotations_out_dir, os.path.basename(annotation_path))
                new_img_path = os.path.join(images_out_dir, os.path.basename(img_path))
                shutil.copy(img_path, new_img_path)
                cv2.imwrite(new_mask_path, new_mask)


cocostuff_water_idx = [178, 179, 155, 148]

find_images_with_label_idx(
    idx_want=cocostuff_water_idx, 
    anns_path=cocostuff164k_annotations_val, 
    imgs_path=cocostuff164k_images_val, 
    annotations_out_dir='cocostuff_water/annotations/validation',
    images_out_dir='cocostuff_water/images/validation')

find_images_with_label_idx(
    idx_want=cocostuff_water_idx, 
    anns_path=cocostuff164k_annotations_train, 
    imgs_path=cocostuff164k_images_train, 
    annotations_out_dir='cocostuff_water/annotations/training',
    images_out_dir='cocostuff_water/images/training')
