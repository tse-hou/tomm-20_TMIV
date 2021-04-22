# train model
import sys
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D



def train_model(T_data_path,T_label_path,V_data_path,V_label_path,testing_dataset):
    T_data = np.load(T_data_path)
    T_label = np.load(T_label_path)
    V_data = np.load(V_data_path)
    V_label = np.load(V_label_path)
    

    # Model

    model = Sequential()
    model.add(keras.layers.Conv2D(128,[8,8],strides=(2,2),padding='same',input_shape=(256,256,56)))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(128,[3,3],strides=(1,1),padding='same'))
    model.add(keras.layers.Activation('relu'))
    # 
    model.add(keras.layers.Conv2D(256,[8,8],strides=(2,2),padding='same'))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(256,[3,3],strides=(1,1),padding='same'))
    model.add(keras.layers.Activation('relu'))
    # 
    model.add(keras.layers.Conv2D(512,[4,4],strides=(2,2),padding='same'))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(512,[3,3],strides=(1,1),padding='same'))
    model.add(keras.layers.Activation('relu'))
    # 
    model.add(keras.layers.Conv2D(512,[4,4],strides=(2,2),padding='same'))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(512,[3,3],strides=(1,1),padding='same'))
    model.add(keras.layers.Activation('relu'))
    # 
    model.add(keras.layers.Conv2D(512,[4,4],strides=(2,2),padding='same'))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(512,[3,3],strides=(1,1),padding='same'))
    model.add(keras.layers.Activation('relu'))
    # 
    model.add(Flatten())
    model.add(keras.layers.Dense(512,activation='relu'))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dense(512,activation='relu'))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dense(512,activation='relu'))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dense(3,activation='sigmoid'))

    adam = keras.optimizers.Adam(0.000001)
    model.compile(loss='mean_squared_error',optimizer=adam,metrics=['mean_squared_error'])
    # model.summary()
    # Tensorboard
    logdir = 'CNN_model/log/log_CNNMM'
    tensorboard_callback = keras.callbacks.TensorBoard(logdir, histogram_freq=1)
    Earlystop = keras.callbacks.EarlyStopping(monitor='val_loss',patience=10)
    model_save = keras.callbacks.ModelCheckpoint(f'CNN_model/model/CNN_{testing_dataset}_20201202_3.h5',monitor='val_loss',save_best_only=True)
    # Fit the model
    history = model.fit(T_data, T_label, 
                        epochs=500,
                        batch_size=32,
                        verbose=1,
                        validation_data=(V_data,V_label),
                        callbacks=[tensorboard_callback,Earlystop,model_save],
                        shuffle=True)
    # model.fit(T_data,T_label,epochs=1000,batch_size=140)
    mse=model.evaluate(T_data,T_label)
    print(mse)

    # evaluation
    # Fit the model
    # list all data in history
    print(history.history.keys())
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(f'CNN_model/result/CNN_{testing_dataset}_20201202_3.png')


if __name__ == "__main__":
    '''Set GPU'''
    gpus = [sys.argv[2]] # Here I set CUDA to only see one GPU
    os.environ['CUDA_VISIBLE_DEVICES']=','.join([str(i) for i in gpus])
    # with tf.device('/gpu:2'):
    dataset = sys.argv[1]
    if (dataset != "ERP"):
        T_data_path = f"CNN_dataset/Training/Training_data_{dataset}.npy"
        T_label_path = f"CNN_dataset/Training/Training_label_{dataset}.npy"
        V_data_path = "CNN_dataset/Testing/Testing_data.npy"
        V_label_path = "CNN_dataset/Testing/Testing_label.npy"
        train_model(T_data_path,T_label_path,V_data_path,V_label_path,dataset)
    elif(dataset == "ERP"):
        T_data_path = f"CNN_dataset/Training/Training_data_NSV.npy"
        T_label_path = f"CNN_dataset/Training/Training_label_NSV.npy"
        V_data_path = "CNN_dataset/Testing/Testing_data_RND.npy"
        V_label_path = "CNN_dataset/Testing/Testing_label_RND.npy"
        train_model(T_data_path,T_label_path,V_data_path,V_label_path,dataset)


