import tensorflow as tf
import numpy as np
import cv2
import os
import random
import threading
import time
from fileinput import input
from tensorflow.core.protobuf.tpu.optimization_parameters_pb2 import LearningRate
import matplotlib.pyplot as plt
from resnet import *
from datagu import *
import tensorflow_addons as tfa

lock = threading.Lock()

DATA_PATH = 'C:/processed_data/train_set'
VALIDATION = 'C:/processed_data/validation_set/validation_set'
CLUSTER_SIZE = 2
CLUSTER_NUM = 100
MAX_QUEUE = 3
FEATURE_NUMS = 32
EPOCH = 1000000
UPDATE_FREQ = 1
SAVE_FREQ = 1000
DIV = 8
ALPHA = 0.2
CONS_S = 2.0


data_queue = []


def load_imgs(path, contrast=True):
    datasets = []
    humans = os.listdir(path)
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
            if(contrast):
                human_datas_app(data[:,::-1,:])
        datasets_app(human_datas)
        max_size = max(max_size, len(human_datas))
    return (datasets, max_size)




def make_dataset(datasets, max_size):
    global data_queue
    while(True):
        max_size = (max_size + CLUSTER_SIZE - 1)//CLUSTER_SIZE*CLUSTER_SIZE
        cluster_num = len(datasets)
        human_seqs = []
        seqs = [i for i in range(cluster_num)]
        for i in range(cluster_num):
            cluster_size = len(datasets[i])
            k = 0
            temps = []
            temp = [j for j in range(cluster_size)]
            while(k <= max_size-cluster_size):
                random.shuffle(temp)
                temps+=temp
                k += cluster_size
            temp = random.sample(temp, max_size-k)
            temps+=temp
            human_seqs.append(temps)
        random.shuffle(seqs)
        
        
        batch_count = 0
        start_idx = 0
        temp_batch = np.empty([CLUSTER_SIZE*CLUSTER_NUM,64,64,3])
        while(start_idx < max_size-1):
            for i in range(cluster_num//CLUSTER_NUM*CLUSTER_NUM):
                for j in range(CLUSTER_SIZE):
                    temp_batch[batch_count,:,:,:] = datasets[seqs[i]][ human_seqs[seqs[i]][start_idx+j]]
                    batch_count += 1
                if(batch_count >= CLUSTER_SIZE*CLUSTER_NUM):
                    while(True):
                        lock.acquire()
                        if(len(data_queue) >= MAX_QUEUE):
                            lock.release()
                            time.sleep(0.1)
                            continue
                        data_queue.append(temp_batch)
                        lock.release()
                        temp_batch = np.empty([CLUSTER_SIZE*CLUSTER_NUM,64,64,3])
                        batch_count = 0
                        break
            start_idx += CLUSTER_SIZE
                    
       


def get_losses(outputs):
    re_outputs = tf.reshape(outputs, shape=[CLUSTER_NUM, CLUSTER_SIZE, FEATURE_NUMS])
    
    
    output_length = tf.sqrt(tf.reduce_sum(tf.square(outputs), axis=1, keepdims=True))
    output_mat = tf.matmul(output_length, output_length, transpose_b=True)
    output_cos = tf.matmul(outputs, outputs, transpose_b=True)/output_mat
    output_cos_exp = tf.reshape(tf.math.exp(output_cos*CONS_S), [CLUSTER_NUM, CLUSTER_SIZE,CLUSTER_NUM, CLUSTER_SIZE])
    output_cos_exp_sum = tf.reduce_sum(tf.reduce_sum(output_cos_exp, axis=1), axis=2)
    softmaxs_diag = tf.linalg.diag_part(output_cos_exp_sum)
    variance_loss = tf.reduce_mean(-tf.math.log(softmaxs_diag/tf.reduce_sum(output_cos_exp_sum, axis=1)))
    
        
    anti_zero_loss = tf.reduce_mean(tf.square(1-tf.abs(outputs)))
    
    
    return (variance_loss,  anti_zero_loss)



def model():
    input1s = tf.keras.Input(shape=(64,64,3,))
    input2s = tf.keras.Input(shape=(64,64,3,))
    res50 = get_resnet18_gauss_f(input1s, 0.0001, 0.001)
    res50.summary()
    dense = tf.keras.layers.Dense(1000, activation='relu')(res50(input2s))
    dense = tf.keras.layers.Dense(1000, activation='relu')(dense)
    dense = tf.keras.layers.Dense(FEATURE_NUMS, activation='tanh')(dense)
    return tf.keras.Model(inputs=input2s, outputs=dense)




grad = 0
def train_step(data, flag):
    global grad
    with tf.GradientTape(persistent=False) as tape:
        outputs = hashModel(data, training=True)
        variance_loss, anti_zero_loss = get_losses(outputs)
        total_loss = variance_loss + ALPHA*anti_zero_loss
    
    if(grad == 0):
        grad = tape.gradient(total_loss, hashModel.trainable_variables)
    else:
        grad += tape.gradient(total_loss, hashModel.trainable_variables)
        
    if(flag):
        opt.apply_gradients(zip(grad, hashModel.trainable_variables))
        grad = 0
    return (variance_loss, anti_zero_loss, total_loss)

total_loss = 10
valid_loss = 1

def train():
    global total_loss
    global valid_loss
    batch = None
    for i in range(EPOCH):
        while(True):
            lock.acquire()
            if(len(data_queue) == 0):
                lock.release()
                time.sleep(0.1)
                continue
            break
        batch = data_queue.pop()
        lock.release()
        batch = data_agument(batch, postAguModel)
        vl, azl, tl = train_step(batch, i%UPDATE_FREQ == UPDATE_FREQ-1)
        print('train|||vl:{:.4f}, azl: {:.4f}, tl: {:.4f}, epoch: {}'.format(vl, azl, tl, i))
        vl, azl, tl = validation(validations)
        print('valid|||vl:{:.4f}, azl: {:.4f}, tl: {:.4f}, epoch: {}'.format(vl, azl, tl, i))
        print('minimum total loss : {}/{}'.format(total_loss,valid_loss))
        if(total_loss > tl and i > 10000):
            valid_loss = vl
            total_loss = tl
            manager1.save()
        if(i%SAVE_FREQ==SAVE_FREQ-1):
            manager2.save()
            
        
    
    

    
def load_validations(path):
    datas,_ = load_imgs(path, contrast=False)    
    ret = np.empty([200, 64,64,3])
    
    idx = 0
    for data_list in random.sample(datas, 100):
        for data in random.sample(data_list, 2):
            ret[idx,:,:,:] = data
            idx+=1
    return ret/127.5-1
    
def validation(data):
    outputs = hashModel(data, training=False)
    variance_loss, anti_zero_loss = get_losses(outputs)
    total_loss = variance_loss + anti_zero_loss*ALPHA
    return (variance_loss, anti_zero_loss, total_loss)
        
        
if __name__ == '__main__':
    random.seed(1234)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
      except RuntimeError as e:
        print(e)
    
    os.environ [ "TF_FORCE_GPU_ALLOW_GROWTH" ] = "true"
        
        
    print('----- loading datasets... -----')
    datasets, num = load_imgs(DATA_PATH)
    validations = load_validations(VALIDATION)
    #print(validations)
    
    
    print('----- make model... -----')
    hashModel = model()
    postAguModel = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.07, fill_mode='constant'),
        tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset=-1),
        tf.keras.layers.experimental.preprocessing.Resizing(64,64)
        ])
    
    
    learning_rate = 1e-4
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    hashModel.compile(optimizer=opt)
    
    checkpoint_dir = 'res18_aug_loss15_l1_feature32_model5_datafixed_cn100_softmax_0.2_2.0'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(hashModel = hashModel,
                                     opt = opt
                                     )
    manager1 = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=5)
    manager2 = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir + "/check", max_to_keep=50)

    
    t1 = threading.Thread(target=make_dataset, args=(datasets, num))
    t2 = threading.Thread(target=train)
    
    t1.start()
    t2.start()
    
    
    
    
    
    
        
        
        
        
        
        
        
    