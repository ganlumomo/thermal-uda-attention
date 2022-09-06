# thermal-uda-cls

## Install

```bash
$ git clone https://github.com/ganlumomo/thermal-uda-cls.git
$ cd thermal-uda-cls
$ conda env create -f environment.yml
$ conda activate thermal-uda-cls
```

## Dataset Preparation

### Download

- MS-COCO:
  - Train set: [http://images.cocodataset.org/zips/train2017.zip](http://images.cocodataset.org/zips/train2017.zip)
  - Val set: [http://images.cocodataset.org/zips/val2017.zip](http://images.cocodataset.org/zips/val2017.zip)
  - Annotations: [http://images.cocodataset.org/annotations/annotations_trainval2017.zip](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
- FLIR:
  - Official: [https://www.flir.com/oem/adas/adas-dataset-form/](https://www.flir.eu/oem/adas/adas-dataset-form/)
  - Kaggle: [https://www.kaggle.com/datasets/deepnewbie/flir-thermal-images-dataset](https://www.kaggle.com/datasets/deepnewbie/flir-thermal-images-dataset)
- M3FD: [https://github.com/JinyuanLiu-CV/TarDAL](https://github.com/JinyuanLiu-CV/TarDAL)

### Process

```bash
$ python utils/prepare_dataset_mscoco_flir.py
$ python utils/prepare_dataset_mscoco_m3fd.py
```

## Running

### Training for MS-COCO to FLIR
```bash
(thermal-uda-cls) $ python core/main.py \
 --tgt_cat flir --n_classes 3 \
 --batch_size 32 --epochs 15 \
 --device cuda:0 --logdir outputs/flir
```

### Training for MS-COCO to M3FD
```bash
(thermal-uda-cls) $ python core/main.py \
 --tgt_cat m3fd --n_classes 6 \
 --batch_size 32 --epochs 30 \
 --device cuda:0 --logdir outputs/m3fd
```

optional:
- ```--self_train```: self training using pseudo labels
- ```--wandb```: enable [wandb](https://wandb.ai/site) logging

### Test
```bash
(thermal-uda-cls) $ python core/test.py \
 --tgt_cat m3fd --n_classes 6 \
 --trained outputs/m3fd/best_model.pt \
 --device cuda:0 --logdir outputs/m3fd
```

optional:
- ```--d_trained outputs/m3fd/best_model_d.pt```: pseudo label generation
- ```--tsne```: enable t-SNE visualization

## Acknowledgement

This repo is based on:
- [SGADA](https://github.com/avaapm/SGADA)
- [ASTMT](https://github.com/facebookresearch/astmt)
