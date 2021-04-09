import torchvision
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from skimage.feature import hog
import time
from skimage import io
from PIL import Image
import cv2

def svm_hog_c(x_train, x_test, y_train, y_test, kernel):
    images_hog_train = hog_c(x_train)
    print(images_hog_train.shape)
    images_hog_test = hog_c(x_test)
    svm_c(images_hog_train, images_hog_test, y_train, y_test, kernel, 'svm_hog_c')

def hog_c(images):
    hog_images = np.empty(shape=[0, 32, 32])
    print(hog_images.shape)

    for i in range(0, len(images)):
        normalised_blocks, hog_image = hog(images[i], orientations=9, pixels_per_cell=(4, 4), cells_per_block=(2, 2), block_norm='L2-Hys',visualize=True)
        # print(hog_image.shape, type(hog_image))
        hog_images = np.append(hog_images, hog_image.reshape((1, 32, 32)), axis=0)

    return hog_images.reshape((len(images), 1024)).copy()

def svm_c(x_train, x_test, y_train, y_test, kernel, name='svm_c'):
    # rbf核函数，设置数据权重
    svc = SVC(class_weight='balanced',)
    c_range = np.logspace(-5, 5, 5, base=2)
    gamma_range = np.logspace(-5, 5, 5, base=2)
    # 网格搜索交叉验证的参数范围，cv=3,3折交叉
    # param_grid = [{'kernel': kernel, 'C': c_range, 'gamma': gamma_range}]
    param_grid = [{'kernel': kernel, 'C': [0.03], 'gamma': [0.001]}]
    grid = GridSearchCV(svc, param_grid, cv=3, n_jobs=-1)
    # 训练模型
    clf = grid.fit(x_train, y_train)
    print(grid.param_grid)
    print('best param:', grid.best_params_)
    print('best_train_score:', grid.best_score_)
    # 计算测试集精度
    score = grid.score(x_test, y_test)
    print(name,'的精度为%s' % score)

if __name__ == '__main__':
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False)
    cifar_train = trainset.data.reshape((50000, 3072))
    cifar_train_label = trainset.targets
    cifar_test = testset.data.reshape((10000, 3072))
    cifar_test_label = testset.targets

    t1 = time.clock()
    svm_c(cifar_train[:1000], cifar_test[:200], cifar_train_label[:1000], cifar_test_label[:200], ['linear'])
    t2 = time.clock()
    print('svm_c_time:', t2-t1)

    t1 = time.clock()
    svm_hog_c(trainset.data[:1000],testset.data[:200], cifar_train_label[:1000], cifar_test_label[:200], ['linear'])
    t2 = time.clock()
    print('svm_c_time:', t2 - t1)