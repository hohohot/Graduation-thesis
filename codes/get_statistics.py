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


DATA_PATH = 'C:/processed_data/test_set'
BATCH_SIZE = 30
FEATURE_NUMS = 8
TARGET_DIR = 'C:/clustering'


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


def load_imgs_part(path):
    datasets = []
    humans = random.sample(os.listdir(path), 990)
    print(humans[2])
    datasets_app = datasets.append
    max_size = 0 
    for human in humans:
        human_datas = []
        human_datas_app = human_datas.append
        human_path = os.path.join(path, human)
        imgs = random.sample(os.listdir(human_path),5)
        for img in imgs:
            img_path = os.path.join(human_path, img)
            data = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)
            data = data
            human_datas_app(data)
            #human_datas_app(data[:,::-1,:])
        datasets_app(human_datas)
        max_size = max(max_size, len(human_datas))
    return (datasets, max_size, humans)



def model():
    input1s = tf.keras.Input(shape=(64,64,3,))
    input2s = tf.keras.Input(shape=(64,64,3,))
    #res50 = get_resnet18_gauss_f(input1s, 0.05, 0.001)
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
    data = np.empty(shape = [5,64,64,3])
    for i in range(5):
        data[i,:,:,:] = datasets[idx][i]
    return data/127.5-1
    
    

def toHash(feature):
    retVal = 0
    for i in range(FEATURE_NUMS):
        retVal = retVal<<1 | int(feature[i])
    return retVal

        
def process_one(idx, offset):
    routputs = test(make_data(idx))
    return (routputs, tf.reduce_sum(routputs, axis=0))
    
    
    
if __name__ == '__main__':
    random.seed(1234)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
      except RuntimeError as e:
        print(e)
        
        
    print('----- loading datasets... -----')
    #datasets, num = load_imgs(DATA_PATH)
    datasets, num, humans = load_imgs_part(DATA_PATH)
    data_len = len(datasets)
    print('----- make model... -----')
    hashModel = model()
    
    
    learning_rate = 1e-4
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(hashModel = hashModel,
                                     opt = opt
                                     )
    
    checkpoint.restore('./res18_aug_loss15_l1_feature2_model5_datafixed_cn100_softmax_0.2_1.0/ckpt-161')
    #checkpoint.restore('./fixed/new old/ckpt-31')
    #checkpoint.restore('./alpha_10_152/ckpt-6')
    
    createFolder(TARGET_DIR)
    
    
    
    
    result=0
    avg = 0
    offset = 0
    output_list = []
    for i in range(len(datasets)):
        print(i)
        r1, r2 = process_one(i, offset)
        output_list.append(r1)
    
    datas_norm = np.concatenate(output_list, axis=0)
    '''datas_norm = datas_norm - datas_norm.mean(axis=0) 
    datas_norm = datas_norm/datas_norm.std(axis=0)'''

    datas_norm[datas_norm>=0] = 1
    datas_norm[datas_norm<0] = 0
    r_datas_norm = np.reshape(datas_norm, (datas_norm.shape[0], 1, FEATURE_NUMS))
    
    positive = {}
    negative = {}
    
    for i in range(datas_norm.shape[0]//5):
        idata = np.reshape(datas_norm[5*i:5*(i+1),:], (1,5,FEATURE_NUMS))
        result = np.abs(r_datas_norm-idata).sum(2)
        primes = result[5*i:5*(i+1),:]
        others = np.concatenate([result[:5*i,:], result[5*(i+1):,:]])
        prime_count = np.unique(primes, return_counts=True)
        other_count = np.unique(others, return_counts=True)
        for pair in zip(prime_count[0], prime_count[1]):
            positive[pair[0]] = positive.get(pair[0], 0) + pair[1]
        for pair in zip(other_count[0], other_count[1]):
            negative[pair[0]] = negative.get(pair[0], 0) + pair[1]
        positive[0] = positive[0]-5
    print(positive)
    print('0-diff : {}, 1-diff: {}, 2-diff: {}'.format((positive[0]/(990*20)),
                                                        ((positive[0]+positive[1])/(990*20)),
                                                        ((positive[0]+positive[1]+positive[2])/(990*20))
                                                        ))
    print('0-diff : {}, 1-diff: {}, 2-diff: {}'.format((negative[0]/(990*25*989)),
                                                        ((negative[0]+negative[1])/(990*25*989)),
                                                        ((negative[0]+negative[1]+negative[2])/(990*25*989))
                                                        ))
    print(negative)
    
    
    
    
        
    
