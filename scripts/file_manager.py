import csv
import shutil
import numpy as np


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


def get_training_set(tag):
    # read labeled training data
    with open('/home/xihan/Myworkspace/TNSCUI2020_train/train.csv', 'r') as file:
        reader = csv.reader(file)
        r = 0
        for row in reader:
            if row[0] != 'ID':
                tag[r, 0] = remove_ext(row[0])
                tag[r, 1] = row[1]
                r = r + 1
    return tag


def cp2train(tag):
    # copy labeled data to a separate folder
    for i in range(len(tag)):
        curr_dir = '/home/xihan/Myworkspace/TNSCUI2020_train/image/{}.PNG'.format(
            int(tag[i, 0]))
        target_dir = '/home/xihan/Myworkspace/TNSCUI2020_train/training_imgs/{}.PNG'.format(
            int(tag[i, 0]))
        shutil.copyfile(curr_dir, target_dir)
        print("copying", curr_dir, "to", target_dir)


def main():
    num_training = 3644     # number of training set
    tag = np.zeros((num_training, 2))
    get_training_set(tag)
    tag = tag[tag[:, 0].argsort()]  # sort tag according to image id
    print(tag[0, 0], tag[0, 1])


if __name__ == "__main__":
    main()
