'''
create input data pipeline
'''
import tensorflow
from misc.utils import *
import pandas as pd
import numpy as np
from collections import defaultdict
from skimage.io import imread
from skimage.transform import resize
import pickle


train_data_csv = '../dataset/train.csv'
aug_data_csv = '../dataset/aug.csv'
train_data_dir = '../dataset/train'
aug_data_dir = '../dataset/aug'
test_data_dir = '../dataset/test'
output_file = 'index_id_mapping.pkl'
batch_size = 32

def id_index_mapping(train_data,output_file):
    x = pd.factorize(train_data.Id)
    mapping = dict(zip(x[1],range(len(x[1]))))
    f = open(output_file,'wb')
    pickle.dump(mapping,f)
    train_data['Id'] = x[0]
    return train_data

def read_from_csv(train_data_csv, aug_data_csv=None, augmentaion=False, other_whale='new_whale'):
    train_data = pd.read_csv(train_data_csv)
    train_data['Image'] = train_data['Image'].apply(lambda x: '../dataset/train/'+x)
    if other_whale:
        train_data.drop(train_data[train_data['Id'].str.contains(other_whale)].index, inplace=True)
    if augmentaion:
        aug_data = pd.read_csv(aug_data_csv)
        aug_data['Image'] = aug_data['Image'].apply(lambda x: '../dataset/aug/'+x)
        train_data = pd.concat([train_data,aug_data])
    train_data = id_index_mapping(train_data, output_file)
    return train_data

def img_process(img_path):
    mean = [103.939, 116.779, 123.68]
    img = imread(img_path)
    img = resize(img, (224, 224))*255.0
    if len(img.shape) == 2:
        img = np.dstack([img,img,img])
    img[:,:,0] -= mean[2]
    img[:,:,1] -= mean[1]
    img[:,:,2] -= mean[0]
    img[:,:,[0,1,2]] = img[:,:,[2,1,0]]
    img = np.reshape(img,[224,224,3])
    return img

def train_input_fn(train_data_csv, batch_size=32, aug_data_csv=None, augmentaion=False, other_whale='new_whale'):
    '''
    :param train_data_csv: path to train data dict
    :param params: hyperparameter
    :return: batch
    '''
    train_data = read_from_csv(train_data_csv, aug_data_csv, augmentaion, other_whale)
    Id_value = train_data.Id.values
    Image_value = train_data.Image.values
    m = len(train_data)
    while True:
        idx = np.random.randint(m,size=batch_size)
        batch_x_pre = Image_value[idx]
        batch_y = Id_value[idx].reshape(-1,1)
        batch_x = np.array(list(map(img_process,batch_x_pre)))
        yield batch_x,batch_y

if __name__ == "__main__":
    i = 0
    x = train_input_fn(train_data_csv)
    while i < 3:

        [y,y2] = x.__next__()
    #y2 = x.__next__()[1]
        print(y[0][0])
        print(y2[0][0])
        i += 1