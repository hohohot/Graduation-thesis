import tensorflow as tf
import numpy as np
import cv2
import os
import random
import threading
import time
from tensorflow.core.protobuf.tpu.optimization_parameters_pb2 import LearningRate
import matplotlib.pyplot as plt
from keras.backend import shape
from resnet import *


DATA_PATH = 'C:/processed_data/validation_set/validation_set'
BATCH_SIZE = 30
MAX_QUEUE = 4
FEATURE_NUMS = 8
EPOCH = 10000
TARGET_DIR = 'C:/examing8'


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def load_imgs(path):
    datasets = []
    humans = os.listdir(path)
    print(humans[2])
    datasets_app = datasets.append
    max_size = 0 
    for human in humans:
        human_datas = []
        human_datas_app = human_datas.append
        human_path = os.path.join(path, human)
        imgs = os.listdir(human_path)
        for img in imgs:
            img_path = os.path.join(human_path, img)
            data = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)
            data = data
            human_datas_app(data)
            human_datas_app(data[:,::-1,:])
        datasets_app(human_datas)
        max_size = max(max_size, len(human_datas))
    return (datasets, max_size)



def model():
    input1s = tf.keras.Input(shape=(64,64,3,))
    input2s = tf.keras.Input(shape=(64,64,3,))
    res50 = get_resnet18_gauss_f(input1s, 0.0001, 0.001)
    res50.summary()
    dense = tf.keras.layers.Dense(1000, activation='relu')(res50(input2s))
    dense = tf.keras.layers.Dense(1000, activation='relu')(dense)
    dense = tf.keras.layers.Dense(FEATURE_NUMS, activation='tanh')(dense)
    return tf.keras.Model(inputs=input2s, outputs=dense)

    
def test(data):
    outputs = np.empty(shape=[data.shape[0],FEATURE_NUMS])
    for i in range(0, data.shape[0], BATCH_SIZE):
        outputs[i:i+BATCH_SIZE,:] = hashModel(data[i:i+BATCH_SIZE,:])
    return outputs

def make_data(idx):
    data = np.empty(shape = [len(datasets[idx]),64,64,3])
    for i in range(len(datasets[idx])):
        data[i,:,:,:] = datasets[idx][i]
    return data/127.5-1
    

def toHash(feature):
    retVal = 0
    for i in range(FEATURE_NUMS):
        retVal |= int(feature[i])<<i
    return retVal

def showimage(value, idx):
    datasets = make_data(idx)
    outputs = tf.round((test(datasets)+1)/2).numpy()
    print(outputs)
    imgs = (datasets+1)*0.5
    for i in range(outputs.shape[0]):
        if(toHash(outputs[i]) == value):
            cv2.imshow('win', imgs[i,:,:,:])
            key = cv2.waitKey()
        
    
def process(outputs, wide):
    ret = np.empty(outputs.shape)
    for i in range(outputs.shape[0]-wide):
        ret[i] = np.average(outputs[i:i+wide,:], axis=0)
    for i in range(wide):
        ret[outputs.shape[0]-wide+i] = (np.sum(outputs[outputs.shape[0]-wide+i:,:], axis=0) + np.sum(outputs[:i,:], axis=0))/wide
    return ret

def seperate(outputs):
    ret = {}
    for i in range(outputs.shape[0]):
        hash = toHash(outputs[i])
        if(ret.get(hash, None) == None):
            ret[hash] = []
        ret[hash].append(i)
    return ret

def save(sep_outputs, idx, offset):
    for key in sep_outputs.keys():
        createFolder(os.path.join(TARGET_DIR, str(key)))
        for id in sep_outputs[key]:
            _, encoded_img = cv2.imencode('.jpg', datasets[idx][id])
            with open(os.path.join(TARGET_DIR, str(key), str(id+offset)+'.jpg'), mode='w+b') as f:
                encoded_img.tofile(f)
        
        
def process_one(idx, offset):
    outputs = np.round((process(test(make_data(idx)), 1)+1)*0.5)
    sep_outputs = seperate(outputs)
    save(sep_outputs, idx, offset)
    
    
        
        
if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
      except RuntimeError as e:
        print(e)
        
        
    print('----- loading datasets... -----')
    datasets, num = load_imgs(DATA_PATH)
    
    
    print('----- make model... -----')
    hashModel = model()
    
    
    learning_rate = 1e-4
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(hashModel = hashModel,
                                     opt = opt
                                     )
    
    checkpoint.restore('./res18_aug_loss15_l1_feature8_model5_datafixed_cn100_softmax_0.2_2.0_/ckpt-109')
    createFolder(TARGET_DIR)
    
    
    offset = 0
    for i in range(len(datasets)):
        print(i)
        process_one(i, offset)
        offset += len(datasets[i])
    
    
    
    
    
    
    
    
        
        
        
        
        
        
        
    