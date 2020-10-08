import numpy as np
import csv
import time
from skimage.feature import hog
from skimage.transform import resize
from sklearn.metrics import f1_score
from sklearn import svm
import joblib
import cv2


height = 128
width = 128
model_path = '/home/xihan/Myworkspace/TNSCUI2020_train/model/'


def rgb2gray(im):
    gray = im[:, :, 0] * 0.2989 + im[:, :, 1] * 0.5870 + im[:, :, 2] * 0.1140
    return gray


def remove_ext(file_name):
    # remove extention in file name
    file_num = list()
    res = 0
    for i in range(len(file_name)):
        if file_name[i] == '.':
            break
        file_num.append(file_name[i])
    for i in range(len(file_num)):
        res = res + int(file_num[i])*pow(10, len(file_num)-i-1)
    return res


def get_feature(tag, start, end):
    t0 = time.time()
    features = list()
    count = 0
    for i in range(start, end):
        file_name = '/home/xihan/Myworkspace/TNSCUI2020_train/training_imgs/{}.PNG'.format(
            int(tag[i, 0]))
        # print('extracting feature from ', file_name)
        img = cv2.imread(file_name)
        img = resize(img, (height, width))
        gray = rgb2gray(img)
        fd = hog(gray, orientations=12, block_norm='L2', pixels_per_cell=[18, 18], cells_per_block=[1, 1], visualize=False,
                 transform_sqrt=True)
        features.append(fd)
        count = count + 1
    t1 = time.time()
    print('finish extracting features from', count,
          'images in {:0.2f} seconds'.format(t1-t0))
    return features


def load_label(tag):
    # read labeled training data
    with open('/home/xihan/Myworkspace/TNSCUI2020_train/train.csv', 'r') as file:
        reader = csv.reader(file)
        r = 0
        for row in reader:
            if row[0] != 'ID':
                tag[r, 0] = remove_ext(row[0])
                tag[r, 1] = row[1]
                r = r + 1
    print('finish loading labels')
    return tag


def train(tag, num_training):
    features = get_feature(tag, 0, num_training)
    print('start training ...')
    t0 = time.time()
    clf = svm.SVC()
    clf.fit(features, tag[0:num_training, 1])
    t1 = time.time()
    print('finish training in {:0.2f} seconds'.format(t1-t0))
    return clf


def test(clf, tag, num_training):
    features = get_feature(tag, num_training+1, len(tag)-1)
    result = clf.predict(features)
    f1 = f1_score(tag[num_training+1:len(tag)-1, 1], result)
    return f1


def main():
    num_img = 3644          # number of total images
    num_training = 3000     # number of images for training
    print('training image: ', num_training)
    print('testing image: ', num_img-num_training)

    tag = np.zeros((num_img, 2))
    load_label(tag)
    tag = tag[tag[:, 0].argsort()]  # sort tag according to image id
    # print(tag[0, 0], tag[0, 1])

    # clf = joblib.load(model_path+'model1')          # load pre-trained model
    clf = train(tag, num_training)        # train model
    # joblib.dump(clf, model_path + 'model1')          # save model

    accuracy = test(clf, tag, num_training)         # test model
    print('f1 score: ', accuracy)


if __name__ == "__main__":
    main()
