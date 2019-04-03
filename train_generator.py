from __future__ import print_function
import numpy as np
from deepdrug3d import DeepDrug3DBuilder
import os
from keras import callbacks
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.utils import multi_gpu_model
from keras.models import Sequential
from data_generator import DataGenerator
# from valid_generator import V_DataGenerator
from keras.models import load_model
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf

config = tf.ConfigProto(device_count={'GPU': 2, 'CPU': 20})
sess = tf.Session(config=config)
keras.backend.set_session(sess)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def load_list(adenines, others, data_folder, labels):
    data_list = []
    folder_list = os.listdir(data_folder)
    folder_list = [f for f in folder_list if 'rotate_voxel_data' in f]
    folder_list.sort()

    a_f_list = folder_list[0:a_times]
    o_f_list = folder_list[0:o_times]

    numm = 0
    for folder in a_f_list:
        for filename in os.listdir(data_folder + folder):
            ll = filename[0:-4].split('_')
            protein_name = ll[0] + "_" + ll[1]
            # full_path = data_folder + folder + '/' + filename

            if protein_name in adenines:

                data_list.append(filename)
                labels[filename] = 1

                numm = numm + 1
                print(numm, end=' ')
                if numm % 20 == 0:
                    print()

    print('\nthe adenine list done')

    num = 0
    for folder in o_f_list:
        for filename in os.listdir(data_folder + folder):
            ll = filename[0:-4].split('_')
            protein_name = ll[0] + "_" + ll[1]
            # full_path = data_folder + folder + '/' + filename

            if protein_name in others:
                data_list.append(filename)
                labels[filename] = 0

                num = num + 1
                print(num, end=' ')
                if num % 20 == 0:
                    print()

    print('the other list done')
    return data_list


# train_list = []
# valid_list = []
# label_list = []

train_folder = './data_prepare/train/'
valid_folder = './data_prepare/valid/'
output = './save_model/'

adenine_list = 'adenine'
other_list = 'other'
a_times = 1
o_times = 1
batch_size = 64
epoch = 100
lr = 0.00001

labels = {}
cnt = 0

adenines = []
with open(adenine_list) as adenine_in:
    for line in adenine_in.readlines():
        temp = line.replace(' ', '').replace('\n', '')
        adenines.append(temp)
others = []
with open(other_list) as other_in:
    for line in other_in.readlines():
        temp = line.replace(' ', '').replace('\n', '')
        others.append(temp)

train_list = load_list(adenines, others, train_folder, labels)
valid_list = load_list(adenines, others, valid_folder, labels)


partition = {"train": train_list, "validation": valid_list}


# Parameters
train_params = {'dim': (32, 32, 32),
                'n_channels': 14,
                'batch_size': batch_size,
                'n_classes': 2,
                'shuffle': True,
                'path': './data_prepare/train/'}
valid_params = {'dim': (32, 32, 32),
                'n_channels': 14,
                'batch_size': batch_size,
                'n_classes': 2,
                'shuffle': True,
                'path': './data_prepare/valid/'}

# Generators
training_generator = DataGenerator(partition['train'], labels, **train_params)
print("the training data is ready")
validation_generator = DataGenerator(partition['validation'], labels, **valid_params)
print("the validating data is ready")

model = DeepDrug3DBuilder.build()
# model = multi_gpu_model(model, gpus=2)
print(model.summary())
adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

# We add metrics to get more results you want to see
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])


earlyStopping = EarlyStopping(monitor='val_loss',
                              patience=10,
                              verbose=0,
                              mode='min')
mcp_save = ModelCheckpoint('.mdl_wts.hdf5',
                           save_best_only=True,
                           monitor='val_loss',
                           mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.2,
                                   patience=20,
                                   verbose=1,
                                   epsilon=1e-4,
                                   mode='min')

print("ready to fit generator")
# Train model on dataset
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=epoch,
                    verbose=2,
                    use_multiprocessing=True,
                    #             callbacks=[tfCallBack],
                    callbacks=[earlyStopping, mcp_save, reduce_lr_loss],
                    #                  validation_split=0.25,
                    workers=8)



if output == None:
    model.save('deepdrug3d.h5')
else:
    if not os.path.exists(output):
        os.mkdir(output)
    if os.path.exists('deepdrug3d.h5'):
        os.remove('deepdrug3d.h5')
    model.save(output + 'deepdrug3d.h5')
    model.save_weights(output + 'weights.h5')
    mm = load_model(output + 'deepdrug3d.h5')
    print(mm.summary())
