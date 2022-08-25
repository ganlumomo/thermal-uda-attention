import os
import json
from PIL import Image


def create_dirs(datasetDir):
    class_names = ['bicycle', 'car', 'person']
    for cls_name in class_names:
        if not os.path.exists(os.path.join(datasetDir, f'uda_data/mscoco/train/{cls_name}/')):
            os.makedirs(os.path.join(datasetDir, f'uda_data/mscoco/train/{cls_name}/'))
        if not os.path.exists(os.path.join(datasetDir, f'uda_data/mscoco/val/{cls_name}/')):
            os.makedirs(os.path.join(datasetDir, f'uda_data/mscoco/val/{cls_name}/'))
        if not os.path.exists(os.path.join(datasetDir, f'uda_data/flir/train/{cls_name}/')):
            os.makedirs(os.path.join(datasetDir, f'uda_data/flir/train/{cls_name}/'))
        if not os.path.exists(os.path.join(datasetDir, f'uda_data/flir/val/{cls_name}/')):
            os.makedirs(os.path.join(datasetDir, f'uda_data/flir/val/{cls_name}/'))


def parse_mscoco(datasetDir, annotations, set_type='train'):
    path_to_images = f'mscoco/{set_type}2017'
    c1 = 0
    count = 0
    im_crop = [0,0,0,0]
    for i in annotations['annotations']:
        if i['category_id']==1:
            c1 = c1 + 1
            ID = i['image_id']
            zeros = 12 - len(str(ID))
            zero = '0' * zeros
            im_person= Image.open(os.path.join(datasetDir, '{}/{}{}.jpg'.format(path_to_images, zero, ID)))
            bbox = i['bbox']
            im_crop[0] = i['bbox'][0]
            im_crop[1] = i['bbox'][1]
            height = im_person.size[1]
            width = im_person.size[0]
            if i['bbox'][2] > i['bbox'][3]:
                im_crop[2] = i['bbox'][0] + i['bbox'][2]
                im_crop[3] = i['bbox'][1] + ((i['bbox'][2]+i['bbox'][3])/2)
                im_crop[1] = i['bbox'][1] - ((i['bbox'][2]-i['bbox'][3])/2)
            if i['bbox'][3] > i['bbox'][2]:
                im_crop[2] = i['bbox'][0] + ((i['bbox'][2]+i['bbox'][3])/2)
                im_crop[0] = i['bbox'][0] - ((i['bbox'][3]-i['bbox'][2])/2)
                im_crop[3] = i['bbox'][1] + i['bbox'][3]
            if i['bbox'][2] == i['bbox'][3]:
                im_crop[2] = i['bbox'][0] + i['bbox'][2]
                im_crop[3] = i['bbox'][1] + i['bbox'][3]
            if (im_crop[2] > width) or (im_crop[3]  > height) or (im_crop[0]<0) or (im_crop[1]<0):
                count = count + 1
                continue
            else:
                area = im_person.crop(im_crop)
                area.save(os.path.join(datasetDir, 'uda_data/mscoco/{}/person/{}.jpg'.format(set_type, c1)))
        if i['category_id'] == 2:
            c1 = c1 + 1
            ID = i['image_id']
            zeros = 12 - len(str(ID))
            zero = '0' * zeros
            im_bicycle = Image.open(os.path.join(datasetDir, '{}/{}{}.jpg'.format(path_to_images, zero, ID)))
            bbox=i['bbox']
            im_crop[0] = i['bbox'][0]
            im_crop[1] = i['bbox'][1]
            height = im_bicycle.size[1]
            width = im_bicycle.size[0]
            if i['bbox'][2] > i['bbox'][3]:
                im_crop[2] = i['bbox'][0] + i['bbox'][2]
                im_crop[3] = i['bbox'][1] + ((i['bbox'][2]+i['bbox'][3])/2)
                im_crop[1] = i['bbox'][1] - ((i['bbox'][2]-i['bbox'][3])/2)
            if i['bbox'][3] > i['bbox'][2]:
                im_crop[2] = i['bbox'][0] + ((i['bbox'][2]+i['bbox'][3])/2)
                im_crop[0] = i['bbox'][0] - ((i['bbox'][3]-i['bbox'][2])/2)
                im_crop[3] = i['bbox'][1] + i['bbox'][3]
            if i['bbox'][2] == i['bbox'][3]:
                im_crop[2] = i['bbox'][0] + i['bbox'][2]
                im_crop[3] = i['bbox'][1] + i['bbox'][3]
            if (im_crop[2] > width) or (im_crop[3]  > height) or (im_crop[0]<0) or (im_crop[1]<0):
                count = count + 1
            else:
                area = im_bicycle.crop(im_crop)
                area.save(os.path.join(datasetDir, 'uda_data/mscoco/{}/bicycle/{}.jpg'.format(set_type, c1)))
        if i['category_id'] == 3:
            c1 = c1 + 1
            ID = i['image_id']
            zeros = 12 - len(str(ID))
            zero = '0' * zeros
            im_car = Image.open(os.path.join(datasetDir, '{}/{}{}.jpg'.format(path_to_images, zero, ID)))
            bbox = i['bbox']
            im_crop[0] = i['bbox'][0]
            im_crop[1] = i['bbox'][1]
            height = im_car.size[1]
            width = im_car.size[0]
            if i['bbox'][2] > i['bbox'][3]:
                im_crop[2] = i['bbox'][0] + i['bbox'][2]
                im_crop[3] = i['bbox'][1] + ((i['bbox'][2]+i['bbox'][3])/2)
                im_crop[1] = i['bbox'][1] - ((i['bbox'][2]-i['bbox'][3])/2)
            if i['bbox'][3] > i['bbox'][2]:
                im_crop[2] = i['bbox'][0] + ((i['bbox'][2]+i['bbox'][3])/2)
                im_crop[0] = i['bbox'][0] - ((i['bbox'][3]-i['bbox'][2])/2)
                im_crop[3] = i['bbox'][1] + i['bbox'][3]
            if i['bbox'][2] == i['bbox'][3]:
                im_crop[2] = i['bbox'][0] + i['bbox'][2]
                im_crop[3] = i['bbox'][1] + i['bbox'][3]
            if (im_crop[2] > width) or (im_crop[3]  > height) or (im_crop[0]<0) or (im_crop[1]<0):
                count = count + 1
            else:
                area = im_car.crop(im_crop)
                area.save(os.path.join(datasetDir, 'uda_data/mscoco/{}/car/{}.jpg'.format(set_type, c1)))


def parse_flir_train(datasetDir, annotations):
    c1 = 0
    count = 0
    im_crop = [0,0,0,0]
    for i in annotations['annotations']:
        if i['category_id']==1:
            c1 = c1 + 1
            ID = i['image_id']+1

            zeros = 5 - len(str(ID))
            zero = '0' * zeros
            im_person = Image.open(os.path.join(datasetDir, 'FLIR_ADAS_1_3/train/thermal_8_bit/FLIR_{}{}.jpeg'.format(zero, ID)))
            bbox = i['bbox']
            im_crop[0] = i['bbox'][0]
            im_crop[1] = i['bbox'][1]
            height = im_person.size[1]
            width = im_person.size[0]
            if i['bbox'][2] > i['bbox'][3]:
                im_crop[2] = i['bbox'][0] + i['bbox'][2]
                im_crop[3] = i['bbox'][1] + ((i['bbox'][2]+i['bbox'][3])/2)
                im_crop[1] = i['bbox'][1] - ((i['bbox'][2]-i['bbox'][3])/2)
            if i['bbox'][3] > i['bbox'][2]:
                im_crop[2] = i['bbox'][0] + ((i['bbox'][2]+i['bbox'][3])/2)
                im_crop[0] = i['bbox'][0] - ((i['bbox'][3]-i['bbox'][2])/2)
                im_crop[3] = i['bbox'][1] + i['bbox'][3]
            if i['bbox'][2] == i['bbox'][3]:
                im_crop[2]=i['bbox'][0] + i['bbox'][2]
                im_crop[3]= i['bbox'][1] + i['bbox'][3]
            if (im_crop[2] > width) or (im_crop[3]  > height) or (im_crop[0]<0) or (im_crop[1]<0):
                count = count + 1
                continue
            else:
                area=im_person.crop(im_crop)
                area.save(os.path.join(datasetDir, 'uda_data/flir/train/person/FLIR_{}.jpeg'.format(c1)))
        if i['category_id'] == 2:
            c1 = c1 + 1
            ID = i['image_id']+1
            zeros = 5 - len(str(ID))
            zero = '0' * zeros
            im_bicycle = Image.open(os.path.join(datasetDir, 'FLIR_ADAS_1_3/train/thermal_8_bit/FLIR_{}{}.jpeg'.format(zero, ID)))
            bbox=i['bbox']
            im_crop[0] = i['bbox'][0]
            im_crop[1] = i['bbox'][1]
            height = im_bicycle.size[1]
            width = im_bicycle.size[0]
            if i['bbox'][2] > i['bbox'][3]:
                im_crop[2] = i['bbox'][0] + i['bbox'][2]
                im_crop[3] = i['bbox'][1] + ((i['bbox'][2]+i['bbox'][3])/2)
                im_crop[1] = i['bbox'][1] - ((i['bbox'][2]-i['bbox'][3])/2)
            if i['bbox'][3] > i['bbox'][2]:
                im_crop[2] = i['bbox'][0] + ((i['bbox'][2]+i['bbox'][3])/2)
                im_crop[0] = i['bbox'][0] - ((i['bbox'][3]-i['bbox'][2])/2)
                im_crop[3] = i['bbox'][1] + i['bbox'][3]
            if i['bbox'][2] == i['bbox'][3]:
                im_crop[2] = i['bbox'][0] + i['bbox'][2]
                im_crop[3] = i['bbox'][1] + i['bbox'][3]
            if (im_crop[2] > width) or (im_crop[3]  > height) or (im_crop[0]<0) or (im_crop[1]<0):
                count = count + 1
            else:
                area = im_bicycle.crop(im_crop)
                area.save(os.path.join(datasetDir, 'uda_data/flir/train/bicycle/FLIR_{}.jpeg'.format(c1)))
        if i['category_id'] == 3:
            c1 = c1 + 1
            ID = i['image_id']+1
            zeros = 5 - len(str(ID))
            zero = '0' * zeros
            im_car = Image.open(os.path.join(datasetDir, 'FLIR_ADAS_1_3/train/thermal_8_bit/FLIR_{}{}.jpeg'.format(zero, ID)))
            bbox = i['bbox']
            im_crop[0] = i['bbox'][0]
            im_crop[1] = i['bbox'][1]
            height = im_car.size[1]
            width = im_car.size[0]
            if i['bbox'][2] > i['bbox'][3]:
                im_crop[2] = i['bbox'][0] + i['bbox'][2]
                im_crop[3] = i['bbox'][1] + ((i['bbox'][2]+i['bbox'][3])/2)
                im_crop[1] = i['bbox'][1] - ((i['bbox'][2]-i['bbox'][3])/2)
            if i['bbox'][3] > i['bbox'][2]:
                im_crop[2] = i['bbox'][0] + ((i['bbox'][2]+i['bbox'][3])/2)
                im_crop[0] = i['bbox'][0] - ((i['bbox'][3]-i['bbox'][2])/2)
                im_crop[3] = i['bbox'][1] + i['bbox'][3]
            if i['bbox'][2] == i['bbox'][3]:
                im_crop[2] = i['bbox'][0] + i['bbox'][2]
                im_crop[3] = i['bbox'][1] + i['bbox'][3]
            if (im_crop[2] > width) or (im_crop[3]  > height) or (im_crop[0]<0) or (im_crop[1]<0):
                count = count + 1
            else:
                area = im_car.crop(im_crop)
                area.save(os.path.join(datasetDir, 'uda_data/flir/train/car/FLIR_{}.jpeg'.format(c1)))


def parse_flir_val(datasetDir, annotations):
    c1 = 0
    count = 0
    im_crop = [0,0,0,0]
    for i in annotations['annotations']:
        if i['category_id']==1:
            c1 = c1 + 1
            ID = i['image_id']+8863

            if (len(str(ID))<5):
                zeros = 5 - len(str(ID))
                zero = '0' * zeros
                im_person = Image.open(os.path.join(datasetDir, 'FLIR_ADAS_1_3/val/thermal_8_bit/FLIR_{}{}.jpeg'.format(zero, ID)))
            elif (len(str(ID))==5):
                im_person = Image.open(os.path.join(datasetDir, 'FLIR_ADAS_1_3/val/thermal_8_bit/FLIR_{}.jpeg'.format(ID)))
            bbox = i['bbox']
            im_crop[0] = i['bbox'][0]
            im_crop[1] = i['bbox'][1]
            height = im_person.size[1]
            width = im_person.size[0]
            if i['bbox'][2] > i['bbox'][3]:
                im_crop[2] = i['bbox'][0] + i['bbox'][2]
                im_crop[3] = i['bbox'][1] + ((i['bbox'][2]+i['bbox'][3])/2)
                im_crop[1] = i['bbox'][1] - ((i['bbox'][2]-i['bbox'][3])/2)
            if i['bbox'][3] > i['bbox'][2]:
                im_crop[2] = i['bbox'][0] + ((i['bbox'][2]+i['bbox'][3])/2)
                im_crop[0] = i['bbox'][0] - ((i['bbox'][3]-i['bbox'][2])/2)
                im_crop[3] = i['bbox'][1] + i['bbox'][3]
            if i['bbox'][2] == i['bbox'][3]:
                im_crop[2] = i['bbox'][0] + i['bbox'][2]
                im_crop[3] = i['bbox'][1] + i['bbox'][3]
            if (im_crop[2] > width) or (im_crop[3]  > height) or (im_crop[0]<0) or (im_crop[1]<0):
                count = count + 1
                continue
            else:
                area = im_person.crop(im_crop)
                area.save(os.path.join(datasetDir, 'uda_data/flir/val/person/FLIR_{}.jpeg'.format(c1)))
        if i['category_id'] == 2:
            c1 = c1 + 1
            ID = i['image_id']+8863
            if (len(str(ID))<5):
                zeros = 5 - len(str(ID))
                zero = '0' * zeros
                im_bicycle = Image.open(os.path.join(datasetDir, 'FLIR_ADAS_1_3/val/thermal_8_bit/FLIR_{}{}.jpeg'.format(zero, ID)))
            elif (len(str(ID))==5):
                im_bicycle = Image.open(os.path.join(datasetDir, 'FLIR_ADAS_1_3/val/thermal_8_bit/FLIR_{}.jpeg'.format(ID)))
            bbox = i['bbox']
            im_crop[0] = i['bbox'][0]
            im_crop[1] = i['bbox'][1]
            height=im_bicycle.size[1]
            width=im_bicycle.size[0]
            if i['bbox'][2] > i['bbox'][3]:
                im_crop[2] = i['bbox'][0] + i['bbox'][2]
                im_crop[3] = i['bbox'][1] + ((i['bbox'][2]+i['bbox'][3])/2)
                im_crop[1] = i['bbox'][1] - ((i['bbox'][2]-i['bbox'][3])/2)
            if i['bbox'][3] > i['bbox'][2]:
                im_crop[2] = i['bbox'][0] + ((i['bbox'][2]+i['bbox'][3])/2)
                im_crop[0] = i['bbox'][0] - ((i['bbox'][3]-i['bbox'][2])/2)
                im_crop[3] = i['bbox'][1] + i['bbox'][3]
            if i['bbox'][2] == i['bbox'][3]:
                im_crop[2] = i['bbox'][0] + i['bbox'][2]
                im_crop[3] = i['bbox'][1] + i['bbox'][3]
            if (im_crop[2] > width) or (im_crop[3]  > height) or (im_crop[0]<0) or (im_crop[1]<0):
                count = count + 1
            else:
                area = im_bicycle.crop(im_crop)
                area.save(os.path.join(datasetDir, 'uda_data/flir/val/bicycle/FLIR_{}.jpeg'.format(c1)))
        if i['category_id'] == 3:
            c1=c1+1
            ID = i['image_id']+8863
            if (len(str(ID))<5):
                zeros = 5 - len(str(ID))
                zero = '0' * zeros
                im_car = Image.open(os.path.join(datasetDir, 'FLIR_ADAS_1_3/val/thermal_8_bit/FLIR_{}{}.jpeg'.format(zero, ID)))
            elif (len(str(ID))==5):
                im_car = Image.open(os.path.join(datasetDir, 'FLIR_ADAS_1_3/val/thermal_8_bit/FLIR_{}.jpeg'.format(ID)))
            bbox = i['bbox']
            im_crop[0] = i['bbox'][0]
            im_crop[1] = i['bbox'][1]
            height=im_car.size[1]
            width=im_car.size[0]
            if i['bbox'][2] > i['bbox'][3]:
                im_crop[2] = i['bbox'][0] + i['bbox'][2]
                im_crop[3] = i['bbox'][1] + ((i['bbox'][2]+i['bbox'][3])/2)
                im_crop[1] = i['bbox'][1] - ((i['bbox'][2]-i['bbox'][3])/2)
            if i['bbox'][3] > i['bbox'][2]:
                im_crop[2] = i['bbox'][0] + ((i['bbox'][2]+i['bbox'][3])/2)
                im_crop[0] = i['bbox'][0] - ((i['bbox'][3]-i['bbox'][2])/2)
                im_crop[3] = i['bbox'][1] + i['bbox'][3]
            if i['bbox'][2] == i['bbox'][3]:
                im_crop[2] = i['bbox'][0] + i['bbox'][2]
                im_crop[3] = i['bbox'][1] + i['bbox'][3]
            if (im_crop[2] > width) or (im_crop[3]  > height) or (im_crop[0]<0) or (im_crop[1]<0):
                count = count + 1
            else:
                area = im_car.crop(im_crop)
                area.save(os.path.join(datasetDir, 'uda_data/flir/val/car/FLIR_{}.jpeg'.format(c1)))


def main():
    datasetDir = os.environ['DATASETDIR']

    print(f'Creating output dirs if not exist under {datasetDir}')
    create_dirs(datasetDir)

    print('Loading MSCOCO training set annotations')
    with open(os.path.join(datasetDir, 'mscoco/annotations_trainval2017/annotations/instances_train2017.json'),'r') as f:
        data = json.load(f)
    print('Parsing MSCOCO training set')
    parse_mscoco(datasetDir, data, set_type='train')

    print('Loading MSCOCO validation set annotations')
    with open(os.path.join(datasetDir, 'mscoco/annotations_trainval2017/annotations/instances_val2017.json'),'r') as f:
        data = json.load(f)
    print('Parsing MSCOCO validation set')
    parse_mscoco(datasetDir, data, set_type='val')

    print('Loading FLIR training set annotations')
    with open(os.path.join(datasetDir, 'FLIR_ADAS_1_3/train/thermal_annotations.json'),'r') as f:
        data = json.load(f)
    print('Parsing FLIR training set')
    parse_flir_train(datasetDir, data)

    print('Loading FLIR validation set annotations')
    with open(os.path.join(datasetDir, 'FLIR_ADAS_1_3/val/thermal_annotations.json'),'r') as f:
        data = json.load(f)
    print('Parsing FLIR validation set')
    parse_flir_val(datasetDir, data)

    print('Done')


if __name__ == '__main__':
    main()
