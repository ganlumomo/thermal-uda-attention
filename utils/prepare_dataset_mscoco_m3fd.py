import os
import json
from PIL import Image
import xml.etree.ElementTree as ET

def create_dirs(datasetDir):
    class_names = ['people', 'car', 'bus', 'motorcycle', 'lamp', 'truck']
    for cls_name in class_names:
        if not os.path.exists(os.path.join(datasetDir, f'uda_data/mscoco2/train/{cls_name}/')):
            os.makedirs(os.path.join(datasetDir, f'uda_data/mscoco2/train/{cls_name}/'))
        if not os.path.exists(os.path.join(datasetDir, f'uda_data/mscoco2/val/{cls_name}/')):
            os.makedirs(os.path.join(datasetDir, f'uda_data/mscoco2/val/{cls_name}/'))
        if not os.path.exists(os.path.join(datasetDir, f'uda_data/m3fd/train/{cls_name}/')):
            os.makedirs(os.path.join(datasetDir, f'uda_data/m3fd/train/{cls_name}/'))
        if not os.path.exists(os.path.join(datasetDir, f'uda_data/m3fd/val/{cls_name}/')):
            os.makedirs(os.path.join(datasetDir, f'uda_data/m3fd/val/{cls_name}/'))
        if not os.path.exists(os.path.join(datasetDir, f'uda_data/m3fd/test/{cls_name}/')):
            os.makedirs(os.path.join(datasetDir, f'uda_data/m3fd/test/{cls_name}/'))

def crop_image(image, j, area_path, count):
    im_width = image.size[0]
    im_height = image.size[1]
    x_min = int(j[4][0].text)
    y_min = int(j[4][1].text)
    x_max = int(j[4][2].text)
    y_max = int(j[4][3].text)
    width = x_max - x_min
    height = y_max - y_min
    im_crop = [x_min, y_min, 0, 0]
    if width > height:
        im_crop[2] = x_min + width
        im_crop[3] = y_min + ((width+height)/2)
        im_crop[1] = y_min - ((width-height)/2)
    if height > width:
        im_crop[2] = x_min + (width+height)/2
        im_crop[0] = x_min - ((height-width)/2)
        im_crop[3] = y_min + height
    if width == height:
        im_crop[2] = x_max
        im_crop[3] = y_max
    if (im_crop[2] > im_width) or (im_crop[3] > im_height) or (im_crop[0] < 0) or (im_crop[1] < 0):
        return count
    else:
        area = image.crop(im_crop)
        area_width = area.size[0]
        area_height = area.size[1]
        #assert area_width == area_height, f"cropped image is not square: width {area_width}, height {area_height}"
        area.save(area_path)
        count = count + 1
    return count

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
                area.save(os.path.join(datasetDir, 'uda_data/mscoco2/{}/people/{}.jpg'.format(set_type, c1)))
        if i['category_id'] == 10:
            c1 = c1 + 1
            ID = i['image_id']
            zeros = 12 - len(str(ID))
            zero = '0' * zeros
            im_lamp = Image.open(os.path.join(datasetDir, '{}/{}{}.jpg'.format(path_to_images, zero, ID)))
            bbox=i['bbox']
            im_crop[0] = i['bbox'][0]
            im_crop[1] = i['bbox'][1]
            height = im_lamp.size[1]
            width = im_lamp.size[0]
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
                area = im_lamp.crop(im_crop)
                area.save(os.path.join(datasetDir, 'uda_data/mscoco2/{}/lamp/{}.jpg'.format(set_type, c1)))
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
                area.save(os.path.join(datasetDir, 'uda_data/mscoco2/{}/car/{}.jpg'.format(set_type, c1)))
        if i['category_id'] == 4:
            c1 = c1 + 1
            ID = i['image_id']
            zeros = 12 - len(str(ID))
            zero = '0' * zeros
            im_motorcycle = Image.open(os.path.join(datasetDir, '{}/{}{}.jpg'.format(path_to_images, zero, ID)))
            bbox = i['bbox']
            im_crop[0] = i['bbox'][0]
            im_crop[1] = i['bbox'][1]
            height = im_motorcycle.size[1]
            width = im_motorcycle.size[0]
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
                area = im_motorcycle.crop(im_crop)
                area.save(os.path.join(datasetDir, 'uda_data/mscoco2/{}/motorcycle/{}.jpg'.format(set_type, c1)))
        if i['category_id'] == 6:
            c1 = c1 + 1
            ID = i['image_id']
            zeros = 12 - len(str(ID))
            zero = '0' * zeros
            im_bus = Image.open(os.path.join(datasetDir, '{}/{}{}.jpg'.format(path_to_images, zero, ID)))
            bbox = i['bbox']
            im_crop[0] = i['bbox'][0]
            im_crop[1] = i['bbox'][1]
            height = im_bus.size[1]
            width = im_bus.size[0]
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
                area = im_bus.crop(im_crop)
                area.save(os.path.join(datasetDir, 'uda_data/mscoco2/{}/bus/{}.jpg'.format(set_type, c1)))
        if i['category_id'] == 8:
            c1 = c1 + 1
            ID = i['image_id']
            zeros = 12 - len(str(ID))
            zero = '0' * zeros
            im_truck = Image.open(os.path.join(datasetDir, '{}/{}{}.jpg'.format(path_to_images, zero, ID)))
            bbox = i['bbox']
            im_crop[0] = i['bbox'][0]
            im_crop[1] = i['bbox'][1]
            height = im_truck.size[1]
            width = im_truck.size[0]
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
                area = im_truck.crop(im_crop)
                area.save(os.path.join(datasetDir, 'uda_data/mscoco2/{}/truck/{}.jpg'.format(set_type, c1)))

def parse_m3df(datasetDir):
    c = [0, 0, 0, 0, 0, 0]
    c_total = [10962, 16210, 551, 477, 2377, 869]
    for i in range(0, 4200):
        annotation = os.path.join(datasetDir, 'M3FD_Detection/Annotation/{file_name:05d}.xml'.format(file_name=i))
        image = Image.open(os.path.join(datasetDir, 'M3FD_Detection/Ir/{file_name:05d}.png'.format(file_name=i)))
        im_width = image.size[0]
        im_height = image.size[1]
        root = ET.parse(annotation).getroot()
        for j in root:
            if j.tag == 'object':
                if j[0].text == 'People':
                    if c[0]/c_total[0] <= 0.8:
                        area_path = os.path.join(datasetDir, 'uda_data/m3fd/train/people/m3fd_{c:05d}.jpeg'.format(c=c[0]))
                    elif c[0]/c_total[0] > 0.8 and c[0]/c_total[0] <= 0.9:
                        area_path = os.path.join(datasetDir, 'uda_data/m3fd/val/people/m3fd_{c:05d}.jpeg'.format(c=c[0]))
                    else:
                        area_path = os.path.join(datasetDir, 'uda_data/m3fd/test/people/m3fd_{c:05d}.jpeg'.format(c=c[0]))
                    c[0] = crop_image(image, j, area_path, c[0])
                if j[0].text == 'Car':
                    if c[1]/c_total[1] <= 0.8:
                        area_path = os.path.join(datasetDir, 'uda_data/m3fd/train/car/m3fd_{c:05d}.jpeg'.format(c=c[1]))
                    elif c[1]/c_total[1] > 0.8 and c[1]/c_total[1] <= 0.9:
                        area_path = os.path.join(datasetDir, 'uda_data/m3fd/val/car/m3fd_{c:05d}.jpeg'.format(c=c[1]))
                    else:
                        area_path = os.path.join(datasetDir, 'uda_data/m3fd/test/car/m3fd_{c:05d}.jpeg'.format(c=c[1]))
                    c[1] = crop_image(image, j, area_path, c[1])
                if j[0].text == 'Bus':
                    if c[2]/c_total[2] <= 0.8:
                        area_path = os.path.join(datasetDir, 'uda_data/m3fd/train/bus/m3fd_{c:05d}.jpeg'.format(c=c[2]))
                    elif c[2]/c_total[2] > 0.8 and c[2]/c_total[2] <= 0.9:
                        area_path = os.path.join(datasetDir, 'uda_data/m3fd/val/bus/m3fd_{c:05d}.jpeg'.format(c=c[2]))
                    else:
                        area_path = os.path.join(datasetDir, 'uda_data/m3fd/test/bus/m3fd_{c:05d}.jpeg'.format(c=c[2]))
                    c[2] = crop_image(image, j, area_path, c[2])
                if j[0].text == 'Motorcycle':
                    if c[3]/c_total[3] <= 0.8:
                        area_path = os.path.join(datasetDir, 'uda_data/m3fd/train/motorcycle/m3fd_{c:05d}.jpeg'.format(c=c[3]))
                    elif c[3]/c_total[3] > 0.8 and c[3]/c_total[3] <= 0.9:
                        area_path = os.path.join(datasetDir, 'uda_data/m3fd/val/motorcycle/m3fd_{c:05d}.jpeg'.format(c=c[3]))
                    else:
                        area_path = os.path.join(datasetDir, 'uda_data/m3fd/test/motorcycle/m3fd_{c:05d}.jpeg'.format(c=c[3]))
                    c[3] = crop_image(image, j, area_path, c[3])
                if j[0].text == 'Lamp':
                    if c[4]/c_total[4] <= 0.8:
                        area_path = os.path.join(datasetDir, 'uda_data/m3fd/train/lamp/m3fd_{c:05d}.jpeg'.format(c=c[4]))
                    elif c[4]/c_total[4] > 0.8 and c[4]/c_total[4] <= 0.9:
                        area_path = os.path.join(datasetDir, 'uda_data/m3fd/val/lamp/m3fd_{c:05d}.jpeg'.format(c=c[4]))
                    else:
                        area_path = os.path.join(datasetDir, 'uda_data/m3fd/test/lamp/m3fd_{c:05d}.jpeg'.format(c=c[4]))
                    c[4] = crop_image(image, j, area_path, c[4])
                if j[0].text == 'Truck':
                    if c[5]/c_total[5] <= 0.8:
                        area_path = os.path.join(datasetDir, 'uda_data/m3fd/train/truck/m3fd_{c:05d}.jpeg'.format(c=c[5]))
                    elif c[5]/c_total[5] > 0.8 and c[5]/c_total[5] <= 0.9:
                        area_path = os.path.join(datasetDir, 'uda_data/m3fd/val/truck/m3fd_{c:05d}.jpeg'.format(c=c[5]))
                    else:
                        area_path = os.path.join(datasetDir, 'uda_data/m3fd/test/truck/m3fd_{c:05d}.jpeg'.format(c=c[5]))
                    c[5] = crop_image(image, j, area_path, c[5])

def main():
    datasetDir = './dataset_dir/'
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

    print('Parsing M3FD data')
    parse_m3df(datasetDir)

    print('Done')


if __name__ == '__main__':
    main()
