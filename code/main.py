import numpy as np
import cv2 as cv
import pickle
import json
import os.path
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from models import WeakClassifier, AdaBoost, Logistic
from utils import compute_all_haar_features, NMS, compute_correct

haar_stride = 12
num_classifiers = 32
model_name = 'logistic'
save_name = ''

sub_windows_size = (6, 8, 10)
window_target_size = (24, 24)
image_target_size = (48, 48)
window_stride = 1
haar_stride = 12

IoU_thershold = 0.5
NMS_IoU_threshold = 0.9
score_threshold = 0.5

path = 'D:/my/study/code/Face_detection/data/'


class Trainer:
    def __init__(self):
        if model_name == 'adaboost':
            self.model = AdaBoost(weak_classifier=WeakClassifier,
                            num_classifiers=num_classifiers)
            self.samples = np.load(path + f'feature/{haar_stride}_stride.npy')
        elif model_name == 'logistic':
            self.model = Logistic()
            self.samples = np.load(path + 'feature/raw.npy')
        self.labels = np.load(path + 'labels.npy')
        
        self.train_mask = np.load(path + 'mask/train_mask.npy')
        self.test_mask = np.load(path + 'mask/test_mask.npy')
        with open(path + 'face_boxes.json') as f:
            self.boxes = json.load(f)
        with open(path + 'test_index.json') as f:
            self.test_index = json.load(f)
        self.test_face_count = 2249

    def train(self):
        self.model.train(self.samples, self.labels,
                         self.train_mask, self.test_mask)

    def detect(self):
        count = 0
        all_face = 0
        t_bar = tqdm(self.test_index)
        for image_index in tqdm(self.test_index):
            correct_count, face = self._detect(image_index)
            count += correct_count
            all_face += face
            t_bar.set_description(
                f'Detection accuracy is {count/all_face:.5f}, all faces: {all_face} ')
            # print(f'Detection accuracy is {count/all_face}, all faces: {all_face} ')
        print(f'Detection accuracy is {correct_count/self.test_face_count}')

    def _detect(self, image_index):
        'Detect single image'
        image, boxes = self._load_image(image_index, image_target_size)
        detected_boxes = []
        scores = []

        for window_size in sub_windows_size:
            for x in range(0, image_target_size[0]-window_size+1, window_stride):
                for y in range(0, image_target_size[1]-window_size+1, window_stride):
                    sub_window = cv.resize(
                        image[x:x+window_size, y:y+window_size], window_target_size)
                    if model_name == 'adaboost':
                        haar_feature = compute_all_haar_features(
                            sub_window, haar_stride).reshape(1, -1)
                        score = self.model.predict_proba(haar_feature)
                    elif model_name == 'logistic':
                        score = self.model.predict_proba(sub_window.reshape(1, -1))
                    if score >= score_threshold:
                        detected_boxes.append(np.array([x, y, x+window_size, y+window_size]))
                        scores.append(score)
        # for window_size in sub_windows_size:
        # for x in range(0, image_target_size[0]-7+1, window_stride):
        #     for y in range(0, image_target_size[1]-10+1, window_stride):
        #         sub_window = cv.resize(
        #             image[x:x+7, y:y+10], window_target_size)
                # if model_name == 'adaboost':
                #     haar_feature = compute_all_haar_features(
                #         sub_window, haar_stride).reshape(1, -1)
                #     score = self.model.predict_proba(haar_feature)
                # elif model_name == 'logistic':
                #     score = self.model.predict_proba(sub_window.reshape(1, -1))
        #         if score >= score_threshold:
        #             detected_boxes.append(np.array([x, y, x+7, y+10]))
        #             scores.append(score)

        detected_boxes = np.array(detected_boxes)
        scores = np.array(scores)
        final_boxes = NMS(detected_boxes, scores, NMS_IoU_threshold)
        correct_count = compute_correct(boxes, final_boxes, IoU_thershold)
        return correct_count, len(boxes)

    def save_model(self, save_name):
        with open(f'../model/{save_name}.pkl', 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self, model_name, stride=12):
        if model_name == 'adaboost':
            with open(f'../model/{model_name}_{stride}stride.pkl', 'rb') as f:
                self.model = pickle.load(f)
        elif model_name == 'logistic':
            with open(f'D:/my/study/code/Face_detection/model/logistic.pkl', 'rb') as f:
                self.model = pickle.load(f)

    def _load_image(self, image_index, target_size=(48, 48)):
        image = cv.imread(cv.samples.findFile(
            path+f'raw_data/Caltech_WebFaces/{image_index}'))
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        height, width = image.shape
        resized_height, resized_width = target_size
        height_scale, width_scale = resized_height/height, resized_width/width
        for id, box in enumerate(self.boxes[image_index]):
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = round(
                x1*height_scale), round(y1*width_scale), round(x2*height_scale), round(y2*width_scale)
            self.boxes[image_index][id] = x1, y1, x2, y2
        image = cv.resize(image, target_size)
        return image, self.boxes[image_index]

if __name__ == '__main__':
    trainer = Trainer()
    # trainer.train()
    trainer.load_model('logistic')
    trainer.detect()
    tra
