#!/usr/bin/env python
# -*- coding: utf8 -*-
import os
from os import listdir
import shutil as sh
import sys
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
import codecs
import cv2
from glob import glob

XML_EXT = ".xml"
ENCODE_METHOD = "utf-8"

############################# Parameter ###########################################
# data_location structure
# | /data_location
#   | images/
#   | labels/
#   | original/

# 한 폴더 안에 이미지(jpg), 라벨(xml), class정보(classes.txt)가 같이 있어야 합니다.

# xml 폴더 경로 작성
dirpath = "./data/images/"  # Absolute Path

label_path = "./data/labels/"  # Absolute Path

# classes.txt 경로 작성
classes_txt = "./classes.txt"  # Absolute Path

# 이미지 형식 작성
ext = ".jpg"

####################################################################################


class PascalVocReader:
    def __init__(self, filepath):
        # shapes type:
        # [labbel, [(x1,y1), (x2,y2), (x3,y3), (x4,y4)], color, color, difficult]
        self.shapes = []
        self.filepath = filepath
        self.verified = False
        try:
            self.parseXML()
        except:
            pass

    def getShapes(self):
        return self.shapes

    def addShape(self, label, bndbox, filename, difficult):
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)
        points = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        self.shapes.append((label, points, filename, difficult))

    def parseXML(self):
        assert self.filepath.endswith(XML_EXT), "Unsupport file format"
        parser = etree.XMLParser(encoding=ENCODE_METHOD)
        xmltree = ElementTree.parse(self.filepath, parser=parser).getroot()
        filename = xmltree.find("filename").text
        path = xmltree.find("path").text
        try:
            verified = xmltree.attrib["verified"]
            if verified == "yes":
                self.verified = True
        except KeyError:
            self.verified = False

        for object_iter in xmltree.findall("object"):
            bndbox = object_iter.find("bndbox")
            label = object_iter.find("name").text
            # Add chris

            difficult = False
            if object_iter.find("difficult") is not None:
                difficult = bool(int(object_iter.find("difficult").text))
            self.addShape(label, bndbox, path, difficult)
        return True


classes = dict()
num_classes = 0

try:
    input = raw_input
except NameError:
    pass

if os.path.isfile(classes_txt):
    # print(classes_txt)
    with open(classes_txt, "r") as f:
        class_list = f.read().strip().split()
        # print(class_list)
        classes = {k: v for (v, k) in enumerate(class_list)}
        # print(classes)


filePaths = glob(dirpath + "/*" + ext)
print(filePaths)
# xmlPaths = glob(dirpath + "/*.xml")
# print(xmlPaths)


for filePath in filePaths:
    # print(filePath)
    filePath = os.path.splitext(filePath)[0]

    with open(filePath + ".txt", "w") as f:
        if not os.path.isfile(filePath + XML_EXT):
            continue

        tVocParseReader = PascalVocReader(filePath + XML_EXT)
        shapes = tVocParseReader.getShapes()

        for shape in shapes:
            class_name = shape[0]
            box = shape[1]
            filename = filePath + ext

            if class_name not in classes.keys():
                classes[class_name] = num_classes
                num_classes += 1
            class_idx = classes[class_name]

            (height, width, _) = cv2.imread(filename).shape

            coord_min = box[0]
            coord_max = box[2]

            xcen = float((coord_min[0] + coord_max[0])) / 2 / width
            ycen = float((coord_min[1] + coord_max[1])) / 2 / height
            w = float((coord_max[0] - coord_min[0])) / width
            h = float((coord_max[1] - coord_min[1])) / height

            f.write("%d %.06f %.06f %.06f %.06f\n" % (class_idx, xcen, ycen, w, h))
            print(class_idx, xcen, ycen, w, h)


with open(dirpath + "/classes.txt", "w") as f:
    for key in classes.keys():
        f.write("%s\n" % key)
        print(key)

os.remove(dirpath + "/classes.txt")
print("\nimages/classes.txt deleted\n")
files = os.listdir(dirpath)


moved_txt = 0
for file in files:
    if "txt" in file.lower() and os.path.isdir(label_path):
        sh.move(dirpath + file, label_path)
        moved_txt += 1

del_check = input("Delete All XML files? : \nY | n  :  ")


if del_check == "y" or del_check == "Y":
    deleted_xml = 0
    for file in files:
        if "xml" in file.lower():
            os.remove(dirpath + file)
            deleted_xml += 1
    print("\n{0} '.xml' files deleted \n\n".format(deleted_xml))

else:
    print("\n XML file didn't removed")
print(
    "{0} '.txt' files moved from /images/ directory to /labels/ directory \n\n".format(
        moved_txt
    )
)
