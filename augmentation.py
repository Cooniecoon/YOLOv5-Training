import imgaug as ia
from imgaug import augmenters as iaa
from pascal_voc_writer import Writer
from os import listdir
import shutil as sh
import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
ia.seed(1)

def read_anntation(xml_file: str):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    bounding_box_list = []

    file_name = root.find('filename').text
    for obj in root.iter('object'):

        object_label = obj.find("name").text
        for box in obj.findall("bndbox"):
            x_min = int(box.find("xmin").text)
            y_min = int(box.find("ymin").text)
            x_max = int(box.find("xmax").text)
            y_max = int(box.find("ymax").text)

        bounding_box = [object_label, x_min, y_min, x_max, y_max]
        bounding_box_list.append(bounding_box)

    return bounding_box_list, file_name

def read_train_dataset(dir):
    images = []
    annotations = []

    for file in listdir(dir):
        if 'jpg' in file.lower() or 'png' in file.lower():
            images.append(cv2.imread(dir + file, 1))
            annotation_file = file.replace(file.split('.')[-1], 'xml')
            bounding_box_list, file_name = read_anntation(dir + annotation_file)
            annotations.append((bounding_box_list, annotation_file, file_name))

    images = np.array(images)

    return images, annotations

def augmentation(images,augmentation,save_dir,filename):
    print('\n',augmentation,'\n')
    for idx in range(len(images)):
        image = images[idx]
        boxes = annotations[idx][0]

        ia_bounding_boxes = []
        for box in boxes:
            ia_bounding_boxes.append(ia.BoundingBox(x1=box[1], y1=box[2], x2=box[3], y2=box[4]))

        bbs = ia.BoundingBoxesOnImage(ia_bounding_boxes, shape=image.shape)

        seq = iaa.Sequential(augmentation)

        seq_det = seq.to_deterministic()

        image_aug = seq_det.augment_images([image])[0]
        bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]

        new_image_file = save_dir + filename + annotations[idx][2]
        cv2.imwrite(new_image_file, image_aug)

        h, w = np.shape(image_aug)[0:2]
        voc_writer = Writer(new_image_file, w, h)

        for i in range(len(bbs_aug.bounding_boxes)):
            bb_box = bbs_aug.bounding_boxes[i]
            voc_writer.addObject(boxes[i][0], int(bb_box.x1), int(bb_box.y1), int(bb_box.x2), int(bb_box.y2))

        voc_writer.save(save_dir + filename + annotations[idx][1])

t_rot=1
t_flip=1    

blur = iaa.AverageBlur(k=(2, 11)) #! 2~11 random
emboss = iaa.Emboss(alpha=(1.0, 1.0), strength=(2.0, 2.0))
gray = iaa.RemoveSaturation(from_colorspace=iaa.CSPACE_BGR)
contrast = iaa.AllChannelsCLAHE(clip_limit=(10, 10), per_channel=True)
bright = iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30))
color = iaa.pillike.EnhanceColor()
sharpen = iaa.Sharpen(alpha=(0.5, 1.0)) #! 0.5 ~ 1.0 random
edge = iaa.pillike.FilterEdgeEnhance()
# about more augmentation parameters: https://imgaug.readthedocs.io/en/latest/index.html

########################################################################################
augmentations=[[blur],[emboss],[gray],[bright],[sharpen,edge]] #! choice augmentation ##
########################################################################################

rotates = [[iaa.Affine(rotate=90)],[iaa.Affine(rotate=180)],[iaa.Affine(rotate=270)]]
flip = iaa.Fliplr(1.0) #! 100% left & right






dir = 'C:/yolo_/aug_test/data/original/' #! Absolute path
save_aug_dir = 'C:/yolo_/aug_test/data/images/' #! Absolute path
##################################################################################################
# Information
aug_len=len(augmentations)
print("\nyou choose {0} augmentation methods\n\n".format(aug_len))

files = os.listdir(dir)


for file in files:
    sh.copy(dir+file, save_aug_dir)
print("{0} files copied from /original/ directory to /images/ directory \n\n".format(len(files)))
##################################################################################################



# Flip
####################################################################################################
images, annotations = read_train_dataset(save_aug_dir)
t_flip=2
augmentation(images, augmentation = flip, save_dir = save_aug_dir , filename = 'fliped_'+str(i)+'_')
print("{0} files created in /images/ directory after flipped\n\n".format((aug_len+1)*len(files)*t_flip))
####################################################################################################



# Rotation 90, 180, 270 degree
####################################################################################################
images, annotations = read_train_dataset(save_aug_dir)
t_rot=4
i=1
for rot in rotates:
    augmentation(images, augmentation = rot, save_dir = save_aug_dir , filename = 'rotated_'+str(i)+'_')
    i+=1
print("{0} files created in /images/ directory after rotated\n\n".format((aug_len+1)*len(files)*t_flip*t_rot))
####################################################################################################



# Augmentation
####################################################################################################
images, annotations = read_train_dataset(save_aug_dir)
i=1
for aug in augmentations:   
    augmentation(images, augmentation = aug, save_dir = save_aug_dir, filename = 'augmented_'+str(i)+'_')
    i+=1
print("{0} files created in /images/ directory after augmented\n\n".format((aug_len)*len(files)))
####################################################################################################  


print("The data has been augmented {0} times ".format((aug_len+1)*t_flip*t_rot))
print("The original image data : {0} images".format(int(len(files)/2)))
print("Total image data : {0} images ".format(int((aug_len+1)*len(files)*t_flip*t_rot/2)))