import os
import numpy as np
import cv2 as cv
from tqdm import tqdm
import cv2 as cv
import numpy as np

### ----------------------------------------------Data Related--------------------------------------------------###

def locate_face(img, coordinates):
    height, width, _ = img.shape
    leye_x, leye_y, reye_x, reye_y, nose_x, nose_y, mouth_x, mouth_y = coordinates
    center_x = nose_x
    center_y = nose_y
    face_h = 2*(mouth_y-(leye_y+reye_y)/2)
    face_w = 2*(reye_x-leye_x)
    top = max(0, round(center_y-(face_h/2)*1.5))
    bottom = min(height, round(center_y+face_h/2))
    left = max(0, round(center_x-face_w/2))
    right = min(width, round(center_x+face_w/2))

    y = top
    x = left
    w = right - left
    h = bottom - top

    coordinates = (x, y, w, h)
    return coordinates


def crop_face(img, coordinates, target_size=(24, 24)):
    x, y, w, h = coordinates
    face = img[y:y+h, x:x+w]
    face = cv.resize(face, target_size)
    return face


def extract_negative_sample(img, face_coordinates, target_size=(24, 24)):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for coordinates in face_coordinates:
        x_min, y_min, x_max, y_max = map(int, coordinates)
        cv.rectangle(mask, (x_min, y_min), (x_max, y_max),
                     (255), thickness=-1)

    contours, _ = cv.findContours(
        mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        if w >= target_size[0] and h >= target_size[1]:
            sample = img[y:y+h, x:x+w]
            resized_sample = cv.resize(sample, target_size)
            return resized_sample
        else:
            break

    start_x = np.random.randint(0, max(1, img.shape[1] - target_size[0]))
    start_y = np.random.randint(0, max(1, img.shape[0] - target_size[1]))
    sample = img[start_y:start_y+target_size[1],
                 start_x:start_x+target_size[0]]
    if sample.shape[0] != target_size[1] or sample.shape[1] != target_size[0]:
        sample = cv.resize(sample, target_size)
    return sample

### ----------------------------------------------Haar Feature Related--------------------------------------------------###


def integral_image(image):
    if image.ndim != 2:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    height, width = image.shape
    integral_image = np.zeros((height + 1, width + 1), dtype=np.float64)
    integral_image[1:, 1:] = image
    integral_image = np.cumsum(np.cumsum(integral_image, axis=0), axis=1)
    return integral_image


def haar_feature_a(integral_image, top_left, bottom_right):
    width = bottom_right[1] - top_left[1]
    height = bottom_right[0] - top_left[0]

    assert width % 2 == 0, "Width should be even."

    left_top = top_left
    left_bottom = (top_left[0] + height, top_left[1] + width // 2)
    right_top = (top_left[0], top_left[1] + width // 2)
    right_bottom = bottom_right

    left_sum = integral_image[left_bottom[0], left_bottom[1]] \
        - integral_image[left_top[0], left_bottom[1]] \
        - integral_image[left_bottom[0], left_top[1]] \
        + integral_image[left_top[0], left_top[1]]

    right_sum = integral_image[right_bottom[0], right_bottom[1]] \
        - integral_image[right_top[0], right_bottom[1]] \
        - integral_image[right_bottom[0], right_top[1]] \
        + integral_image[right_top[0], right_top[1]]
    return right_sum - left_sum


def haar_feature_b(integral_image, top_left, bottom_right):
    width = bottom_right[1] - top_left[1]
    height = bottom_right[0] - top_left[0]

    assert height % 2 == 0, "Height should be even."

    top_part = top_left
    bottom_part = (top_left[0] + height // 2, bottom_right[1])

    top_sum = integral_image[top_part[0] + height//2, top_part[1]] \
        - integral_image[top_part[0], top_part[1]] \
        - integral_image[top_part[0] + height//2, bottom_right[1]] \
        + integral_image[top_part[0], bottom_right[1]]

    bottom_sum = integral_image[bottom_right[0], bottom_right[1]] \
        - integral_image[bottom_part[0], bottom_right[1]] \
        - integral_image[bottom_right[0], bottom_part[1]] \
        + integral_image[bottom_part[0], bottom_part[1]]
    return top_sum - bottom_sum


def haar_feature_c(integral_image, top_left, bottom_right):
    width = bottom_right[1] - top_left[1]
    height = bottom_right[0] - top_left[0]

    assert width % 3 == 0, "Width should be divisible by 3."

    one_third_width = width // 3

    left_sum = integral_image[top_left[0], top_left[1] + one_third_width] \
        - integral_image[top_left[0], top_left[1]] \
        - integral_image[bottom_right[0], top_left[1] + one_third_width] \
        + integral_image[bottom_right[0], top_left[1]]

    middle_sum = integral_image[top_left[0], top_left[1] + 2*one_third_width] \
        - integral_image[top_left[0], top_left[1] + one_third_width] \
        - integral_image[bottom_right[0], top_left[1] + 2*one_third_width] \
        + integral_image[bottom_right[0], top_left[1] + one_third_width]

    right_sum = integral_image[bottom_right[0], bottom_right[1]] \
        - integral_image[bottom_right[0], top_left[1] + 2*one_third_width] \
        - integral_image[top_left[0], bottom_right[1]] \
        + integral_image[top_left[0], top_left[1] + 2*one_third_width]
    return middle_sum - (left_sum + right_sum) / 2


def haar_feature_d(integral_image, top_left, bottom_right):
    width = bottom_right[1] - top_left[1]
    height = bottom_right[0] - top_left[0]

    assert width % 2 == 0 and height % 2 == 0, "Width and height should be even."

    half_width = width // 2
    half_height = height // 2

    top_left_sum = integral_image[top_left[0] + half_height, top_left[1] + half_width] \
        - integral_image[top_left[0], top_left[1] + half_width] \
        - integral_image[top_left[0] + half_height, top_left[1]] \
        + integral_image[top_left[0], top_left[1]]

    bottom_right_sum = integral_image[bottom_right[0], bottom_right[1]] \
        - integral_image[top_left[0] + half_height, bottom_right[1]] \
        - integral_image[bottom_right[0], top_left[1] + half_width] \
        + integral_image[top_left[0] + half_height, top_left[1] + half_width]

    diagonal_sum = top_left_sum + bottom_right_sum

    total_sum = integral_image[bottom_right[0], bottom_right[1]] \
        - integral_image[top_left[0], bottom_right[1]] \
        - integral_image[bottom_right[0], top_left[1]] \
        + integral_image[top_left[0], top_left[1]]
    return 2 * diagonal_sum - total_sum


def generate_all_haar_features(image_shape, stride=1):
    features = []
    height, width = image_shape

    for y in range(0, height, stride):
        for x in range(0, width, stride):
            for h in range(1, height - y + 1):
                for w in range(1, (width - x) // 2 + 1):
                    features.append(('a', (x, y, x + 2*w, y + h)))

    for y in range(0, height, stride):
        for x in range(0, width, stride):
            for h in range(1, (height - y) // 2 + 1):
                for w in range(1, width - x + 1):
                    features.append(('b', (x, y, x + w, y + 2*h)))

    for y in range(0, height, stride):
        for x in range(0, width, stride):
            for h in range(1, height - y + 1):
                for w in range(1, (width - x) // 3 + 1):
                    features.append(('c', (x, y, x + 3*w, y + h)))
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            for h in range(1, (height - y) // 2 + 1):
                for w in range(1, (width - x) // 2 + 1):
                    features.append(('d', (x, y, x + 2*w, y + 2*h)))

    return features


def compute_haar_feature(integral_img, feature):
    feature_type, (x, y, x2, y2) = feature
    top_left = (y, x)
    bottom_right = (y2, x2)

    if feature_type == 'a':
        return haar_feature_a(integral_img, top_left, bottom_right)
    elif feature_type == 'b':
        return haar_feature_b(integral_img, top_left, bottom_right)
    elif feature_type == 'c':
        return haar_feature_c(integral_img, top_left, bottom_right)
    elif feature_type == 'd':
        return haar_feature_d(integral_img, top_left, bottom_right)
    else:
        raise ValueError("Unknown feature type")


def normalize(image, feature_values):
    sigma = 1e-4
    mean = np.mean(image)
    std_dev = np.std(image)
    feature_values = (feature_values-mean)/(std_dev+sigma)
    return feature_values


def compute_all_haar_features(image, stride):
    integral_img = integral_image(image)
    features = generate_all_haar_features(image.shape, stride)
    feature_values = np.array([compute_haar_feature(
        integral_img, feature) for feature in features])
    # feature_values = normalize(image, feature_values)
    return feature_values


### ----------------------------------------------Detection Related--------------------------------------------------###

def IoU(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    x_inter1 = max(x1, x3)
    y_inter1 = max(y1, y3)
    x_inter2 = min(x2, x4)
    y_inter2 = min(y2, y4)

    interArea = max(0, x_inter2-x_inter1+1)*max(0, y_inter2-y_inter1+1)

    area_box1 = (x2-x1+1)*(y2-y1+1)
    area_box2 = (x4-x3+1)*(y4-y3+1)

    IoU = interArea / (area_box1+area_box2-interArea)
    return IoU


def NMS(boxes, scores, threshold):
    if len(boxes) == 0:
        return []
    pick = []
    idxs = np.argsort(scores)
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [j for j in range(len(idxs)) if IoU(boxes[j], boxes[last]) > threshold]
        idxs = np.delete(idxs, suppress)
    return boxes[pick]


def compute_correct(face_boxes, detected_boxes, IoU_thershold):
    count = 0
    for face_box in face_boxes:
        for detected_box in detected_boxes:
            if IoU(face_box, detected_box) >= IoU_thershold:
                count += 1
                break
    return count