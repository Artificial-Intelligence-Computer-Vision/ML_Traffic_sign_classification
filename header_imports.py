# Copyright © 2021 Ronaldson Bellande
from __future__ import print_function
import cv2, sys, math, random, warnings, os, os.path, json, pydicom, glob, shutil, datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from os.path import basename
from PIL import Image
from tensorflow import keras
import tensorflow as tf

from imgaug import augmenters as iaa
from tqdm import tqdm

from os import listdir
from xml.etree import ElementTree
from matplotlib import pyplot
from matplotlib.patches import Rectangle
import coco

from sklearn.tree import DecisionTreeRegressor
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import RobustScaler, StandardScaler, LabelEncoder, MinMaxScaler, OneHotEncoder, LabelBinarizer
from sklearn.metrics import mean_squared_error, accuracy_score, mean_absolute_error,  classification_report, confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, average_precision_score, auc, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, KFold, cross_val_predict, StratifiedKFold, train_test_split, learning_curve, ShuffleSplit, train_test_split,  KFold, cross_val_score, StratifiedShuffleSplit
from sklearn import model_selection, preprocessing
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier
import keras

from mrcnn.config import Config
import mrcnn.model as modellib
from mrcnn.model import MaskRCNN, mold_image, log
from mrcnn.utils import Dataset
from mrcnn import utils, visualize

from keras.datasets import cifar10
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout, Activation
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from tensorflow.keras.utils import to_categorical
import matplotlib.image as img

from contextlib import redirect_stdout
from multiprocessing import Pool
warnings.filterwarnings('ignore')
plt.style.use('ggplot')

from computer_vision_model_building import *
from computer_vision_model_training import *
from computer_vision_model_transfer_learning import *
from computer_vision_model_detection import *
