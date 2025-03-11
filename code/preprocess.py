import os
import numpy as np
import cv2 as cv
import json
from tqdm import tqdm
from utils import locate_face, crop_face, extract_negative_sample, compute_all_haar_features

target_size = (24, 24)
stride = 1

# 提取所有人脸的位置信息并存储
with open('D:\my\study\code\Face_detection\data\WebFaces_GroundThruth.txt', 'r') as f:
    lines = f.readlines()

face_coordinates = {}
image_folder = 'D:\my\study\code\Face_detection\data\Caltech_WebFaces'
for line in tqdm(lines, desc='Computing face coordinates'):
    parts = line.strip().split()
    image_name = parts[0]
    image_path = os.path.join(image_folder, image_name)
    img = cv.imread(image_path)
    coordinates = list(map(float, parts[1:]))
    coordinates = locate_face(img, coordinates)
    if image_name in face_coordinates:
        face_coordinates[image_name].append(coordinates)
    else:
        face_coordinates[image_name] = [coordinates]


samples = []
labels = []
# 提取正负样本
image_folder = 'D:\my\study\code\Face_detection\data\Caltech_WebFaces'
for image_name in tqdm(os.listdir(image_folder), desc='Generating samples and features'):
    image_path = os.path.join(image_folder, image_name)
    img = cv.imread(image_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    for face_coordinate in face_coordinates[image_name]:
        face = crop_face(img, face_coordinate, target_size)
        samples.append(compute_all_haar_features(face, stride=stride))
        labels.append(1)

    negative_sample = extract_negative_sample(img,
                                              face_coordinates[image_name],
                                              target_size)
    samples.append(compute_all_haar_features(negative_sample, stride=stride))
    labels.append(0)

samples = np.array(samples)
labels = np.array(labels)
np.save('D:/my/study/code/Face_detection/data/features', samples)
# np.save('D:/my/study/code/Face_detection/data/labels', labels)
