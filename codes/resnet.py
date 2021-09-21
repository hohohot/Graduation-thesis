import tensorflow as tf

from tensorflow.keras.layers import *



def conv1_layer(x):    
    x = ZeroPadding2D(padding=(2, 2))(x)
    x = Conv2D(64, (5, 5), strides=(1, 1))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPool2D((3,3), 2)(x)
    return x 




def conv_layer(x, shapes, repeats):          
    shortcut = x
 
    for i in range(repeats):
        if (i == 0):
            for shape in shapes[:-1]:
                print(shape)
                print(x.shape)
                x = Conv2D(shape[0], shape[1], strides=(1, 1), padding=shape[2])(x)
                print(x.shape)
                x = BatchNormalization()(x)
                print(x.shape)
                x = Activation('relu')(x)
 
            x = Conv2D(shapes[-1][0], shapes[-1][1], strides=(1, 1), padding=shapes[-1][2])(x)
            shortcut = Conv2D(shapes[-1][0], (1,1), strides=(1, 1), padding='valid')(shortcut)            
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)
            
            print(x.shape)
            print(shortcut.shape)
            x = Add()([x, shortcut])
            x = Activation('relu')(x)
            
            shortcut = x
 
        else:
            for shape in shapes[:-1]:
                x = Conv2D(shape[0], shape[1], strides=(1, 1), padding=shape[2])(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)
 
            x = Conv2D(shapes[-1][0], shapes[-1][1], strides=(1, 1), padding=shapes[-1][2])(x)
            x = BatchNormalization()(x)            
 
            x = Add()([x, shortcut])   
            x = Activation('relu')(x)  
 
            shortcut = x        
    return x

def conv_gaussian_layer(x, shapes, repeats, stddev, reg):          
    shortcut = x
 
    for i in range(repeats):
        if (i == 0):
            for shape in shapes[:-1]:
                print(shape)
                print(x.shape)
                x = Conv2D(shape[0], shape[1], strides=(1, 1), padding=shape[2], kernel_regularizer=tf.keras.regularizers.l2(reg))(x)
                print(x.shape)
                x = BatchNormalization()(x)
                print(x.shape)
                x = GaussianNoise(stddev)(x)
                x = Activation('relu')(x)
 
            x = Conv2D(shapes[-1][0], shapes[-1][1], strides=(1, 1), padding=shapes[-1][2])(x)
            shortcut = Conv2D(shapes[-1][0], (1,1), strides=(1, 1), padding='valid')(shortcut)            
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)
            
            print(x.shape)
            print(shortcut.shape)
            x = Add()([x, shortcut])
            x = GaussianNoise(stddev)(x)
            x = Activation('relu')(x)
            
            shortcut = x
 
        else:
            for shape in shapes[:-1]:
                x = Conv2D(shape[0], shape[1], strides=(1, 1), padding=shape[2], kernel_regularizer=tf.keras.regularizers.l2(reg))(x)
                x = BatchNormalization()(x)
                x = GaussianNoise(stddev)(x)
                x = Activation('relu')(x)
 
            x = Conv2D(shapes[-1][0], shapes[-1][1], strides=(1, 1), padding=shapes[-1][2])(x)
            x = BatchNormalization()(x)            
 
            x = Add()([x, shortcut])   
            x = GaussianNoise(stddev)(x)
            x = Activation('relu')(x)  
 
            shortcut = x        
    return x

def get_resnet18_gauss(input_tensor, stddev, reg):
    x = conv1_layer(input_tensor)
    print(x.shape)
    x = conv_gaussian_layer(x, [[64, (3,3), 'same'], [64, (3,3), 'same']], 2, stddev, reg)
    x = MaxPool2D((2,2), 2)(x)
    x = conv_gaussian_layer(x, [[128, (3,3), 'same'], [128, (3,3), 'same']], 2, stddev, reg)
    x = MaxPool2D((2,2), 2)(x)
    x = conv_gaussian_layer(x, [[256, (3,3), 'same'], [256, (3,3), 'same']], 2, stddev, reg)
    x = MaxPool2D((2,2), 2)(x)
    x = conv_gaussian_layer(x, [[512, (3,3), 'same'], [512, (3,3), 'same']], 2, stddev, reg)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1000)(x)
    return tf.keras.Model(inputs=input_tensor, outputs=x)

def get_resnet18_gauss_f(input_tensor, stddev, reg):
    x = conv1_layer(input_tensor)
    print(x.shape)
    x = conv_gaussian_layer(x, [[64, (3,3), 'same'], [64, (3,3), 'same']], 2, stddev, reg)
    x = MaxPool2D((2,2), 2)(x)
    x = conv_gaussian_layer(x, [[128, (3,3), 'same'], [128, (3,3), 'same']], 2, stddev, reg)
    x = MaxPool2D((2,2), 2)(x)
    x = conv_gaussian_layer(x, [[256, (3,3), 'same'], [256, (3,3), 'same']], 2, stddev, reg)
    x = MaxPool2D((2,2), 2)(x)
    x = conv_gaussian_layer(x, [[512, (3,3), 'same'], [512, (3,3), 'same']], 2, stddev, reg)
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    return tf.keras.Model(inputs=input_tensor, outputs=x)

def get_resnet34_gauss(input_tensor, stddev, reg):
    x = conv1_layer(input_tensor)
    print(x.shape)
    x = conv_gaussian_layer(x, [[64, (3,3), 'same'], [64, (3,3), 'same']], 3, stddev, reg)
    x = MaxPool2D((2,2), 2)(x)
    x = conv_gaussian_layer(x, [[128, (3,3), 'same'], [128, (3,3), 'same']], 4, stddev, reg)
    x = MaxPool2D((2,2), 2)(x)
    x = conv_gaussian_layer(x, [[256, (3,3), 'same'], [256, (3,3), 'same']], 6, stddev, reg)
    x = MaxPool2D((2,2), 2)(x)
    x = conv_gaussian_layer(x, [[512, (3,3), 'same'], [512, (3,3), 'same']], 3, stddev, reg)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1000)(x)
    return tf.keras.Model(inputs=input_tensor, outputs=x)

def get_resnet50_gauss(input_tensor, stddev, reg):
    x = conv1_layer(input_tensor)
    print(x.shape)
    x = conv_gaussian_layer(x, [[64, (1,1), 'valid'], [64, (3,3), 'same'], [256, (1,1), 'valid']], 3, stddev, reg)
    x = MaxPool2D((2,2), 2)(x)
    x = conv_gaussian_layer(x, [[128, (1,1), 'valid'], [128, (3,3), 'same'], [512, (1,1), 'valid']], 4, stddev, reg)
    x = MaxPool2D((2,2), 2)(x)
    x = conv_gaussian_layer(x, [[256, (1,1), 'valid'], [256, (3,3), 'same'], [1024, (1,1), 'valid']], 6, stddev, reg)
    x = MaxPool2D((2,2), 2)(x)
    x = conv_gaussian_layer(x, [[512, (1,1), 'valid'], [512, (3,3), 'same'], [2048, (1,1), 'valid']], 3, stddev, reg)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1000)(x)
    return tf.keras.Model(inputs=input_tensor, outputs=x)


def get_resnet18(input_tensor):
    x = conv1_layer(input_tensor)
    print(x.shape)
    x = conv_layer(x, [[64, (3,3), 'same'], [64, (3,3), 'same']], 2)
    x = MaxPool2D((2,2), 2)(x)
    x = conv_layer(x, [[128, (3,3), 'same'], [128, (3,3), 'same']], 2)
    x = MaxPool2D((2,2), 2)(x)
    x = conv_layer(x, [[256, (3,3), 'same'], [256, (3,3), 'same']], 2)
    x = MaxPool2D((2,2), 2)(x)
    x = conv_layer(x, [[512, (3,3), 'same'], [512, (3,3), 'same']], 2)
    
    x = GlobalAveragePooling2D()(x)
    x = Dense(1000)(x)
    return tf.keras.Model(inputs=input_tensor, outputs=x)

def get_resnet34(input_tensor):
    x = conv1_layer(input_tensor)
    print(x.shape)
    x = conv_layer(x, [[64, (3,3), 'same'], [64, (3,3), 'same']], 3)
    x = MaxPool2D((2,2), 2)(x)
    x = conv_layer(x, [[128, (3,3), 'same'], [128, (3,3), 'same']], 4)
    x = MaxPool2D((2,2), 2)(x)
    x = conv_layer(x, [[256, (3,3), 'same'], [256, (3,3), 'same']], 6)
    x = MaxPool2D((2,2), 2)(x)
    x = conv_layer(x, [[512, (3,3), 'same'], [512, (3,3), 'same']], 3)
    
    x = GlobalAveragePooling2D()(x)
    x = Dense(1000)(x)
    return tf.keras.Model(inputs=input_tensor, outputs=x)


    
