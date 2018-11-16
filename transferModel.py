# This file import createTransferCodes file
# createTransferCodes does following things:
# 1. download the VGG16 model data (.npy file)
# 2. apply transformations if required
# 3. create transfer codes and store them in the project directory

import numpy as np
import csv
import dataHandler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import LearningRateScheduler
from keras import backend as K
from sklearn.metrics import accuracy_score, confusion_matrix
import vgg16


EPOCHS = 50
optimizer = Adam(lr=0.01)
create_new_transfer_codes = True


# ---------------------------------------------------
if create_new_transfer_codes:
    print('Creating new transfer codes for data...')
    import createTransferCodes

# ---------------------------------------------------

# load files
with open('labels') as f:
    reader = csv.reader(f, delimiter='\n')
    labels = np.array([each for each in reader]) #.squeeze()
    print(len(labels),'len 2 labels')
    #labels = labels[:-1]
    print(len(labels),'len 3 labels')
    print('loaded labels', labels.shape)

with open('codes') as f:
    codes = np.fromfile(f, dtype=np.float32)
    print(len(codes),"len of codes file")
    print(codes)
    codes = codes.reshape((len(labels), -1))
    print('loaded codes', codes.shape)

vgg = vgg16.Vgg16()

# -------------------------------------------------------------
# split data
from sklearn.model_selection import train_test_split
labels, classes = dataHandler.one_hot_encode(labels)
X_train, X_test, y_train, y_test = train_test_split(codes, labels, test_size=3776, random_state=42)
X_train = X_train.astype('float32')
print('X shape', X_train.shape)
print('y shape', y_train.shape)

# --------------------------------------------------------------
lr_scheduler = LearningRateScheduler(dataHandler.scheduler)
K.set_image_dim_ordering('th')

# create model
model = Sequential()
model.add(Dense(512, input_shape=(4096,)))
model.add(Activation('tanh'))
model.add(Dropout(0.1))
#model.add(Dense(512))
#model.add(Activation('relu'))
#model.add(Dropout(0.3))
#model.add(Dense(256))
#model.add(Activation('tanh'))
#model.add(Dropout(0.1))
model.add(Dense(classes))
model.add(Activation('softmax'))


# Compile and fit model
# LR scheduler added which updates LR every 10 epochs
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=32, nb_epoch=EPOCHS, validation_data=(X_test, y_test), callbacks=[lr_scheduler])

# predict
result = np.rint(model.predict(X_test))
print(result.shape,"shape of result")
print(y_test.shape,"label shape")
print('predicted', np.argmax(result, axis=1))
print('actual', np.argmax(y_test, axis=1))
print(accuracy_score(y_test, result))
y_test = np.argmax(y_test, axis=1)
result = np.argmax(result, axis=1)
print(result[resulta])
print(confusion_matrix(y_test,result))
#print(result)
# i,res=enumerate(result)

''' Predicting without trainint the VGG model on new data.
    Change createTransferCodes codesbatch = sess.run(vgg.prob, feed_dict=feed_dict) 
    remove the additional layer added in this code to see predictions on org VGG16 model  '''
# conf=np.zeros((8,1000))
# org=np.zeros(8)
# for k in range(3776):
#     conf[np.argmax(y_test[k] )][np.argmax(result[k])]+=1
#     org[np.argmax(y_test[k] )]+=1
#     print(np.argmax(result[k]), np.argmax(y_test[k] ))
# print(org)
# print(np.argmax(conf,axis=1))
# for k in range(8):
#     print(conf[k][np.argmax(conf[k])],conf[k][np.argmax(conf[k])]/org[k],vgg.label_string[np.argmax(conf[k])])


