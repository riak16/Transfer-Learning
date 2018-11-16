import urllib #.request import urlretrieve
from os.path import isfile, isdir
import os
import numpy as np
import tensorflow as tf
#import tensornets as nets
#from tensorflow_vgg import vgg16
#from tensorflow_vgg import util
import utils
import vgg16
#from keras.applications.vgg16 import VGG16
import dataHandler
import csv

vgg_dir = 'tensorflow_vgg/'
data_dir = 'data/'

# -------------------------------------------------------------
# download trained model data
if not isdir(vgg_dir):
    os.makedirs(vgg_dir)
    print('VGG directory created!')

# check if the model trained parameters file is present
if not isfile(vgg_dir + "vgg16.npy"):
    urllib.urlretrieve(
        'https://s3.amazonaws.com/content.udacity-data.com/nd101/vgg16.npy',
        vgg_dir + 'vgg16.npy', dataHandler.reporthook)
else:
    print("Parameter file already exists!")


# --------------------------------------------------------------
# transform images if data is too less
contents = os.listdir(data_dir)
classes = [each for each in contents if os.path.isdir(data_dir + each)]
print('Classes found:', classes)

print('_'*50)
print('Creating image transformations wherever data is less:')
dataHandler.transform_images(data_dir=data_dir, minimum_files_required=1)

# -----------------------------------------------------------
# Create transfer codes for each image and store in files 'codes, labels'
batch_size = 10
codes_list = []
labels = []
batch = []

codes = None

print('_'*50)
print('Creating transfer codes:')
print('_'*50)

with tf.Session() as sess:
    with tf.Session() as sess1:
        #vgg = vgg.vgg16()
        #vgg=VGG16(weights='imagenet')
        vgg = vgg16.Vgg16()
        input_ = tf.placeholder(tf.float32, [None, 224, 224, 3])
        with tf.name_scope("content_vgg"):
            vgg.build(input_)
            #r=tf.nn.relu6
            t1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="relu6")

    for each in classes:
        print("Starting {} images".format(each))
        class_path = data_dir + each
        files = os.listdir(class_path)
	#a=tf.Print(t1,[t1])
        #print(a)
        #print("\n\n\n\n a \n\n\n\n")
        for ii, file in enumerate(files,1):

            # Add images to the current batch
            print(file,"file and class path", class_path)
            img = utils.load_image(os.path.join(class_path, file))
            batch.append(img.reshape((1, 224, 224, 3)))
            labels.append(each)
            #print(len(labels),'labels and number',ii)

            # Running the batch through the network to get the codes
            if ii % batch_size == 0 or ii == (len(files)):

                # Image batch to pass to VGG network
                images = np.concatenate(batch)

                # Get the values from the relu6 layer of the VGG network
                feed_dict = {input_: images}
                codes_batch = sess.run(vgg.relu6, feed_dict=feed_dict)

                # store the codes in an array
                if codes is None:
                    codes = codes_batch
                    print(" Code is none ",ii)
                else:
                    codes = np.concatenate((codes, codes_batch))
                    print(ii," code concat", len(codes))
                print("label-code",len(labels)-len(codes))

                # Reset to start building the next batch
                batch = []
                print('{} images processed'.format(ii))

# -----------------------------------------------------------
# store codes locally
with open('codes', 'w') as f:
    codes.tofile(f)
    print('Transfer codes saved to file "codes" in project directory')

# store labels locally
with open('labels', 'w') as f:
    writer = csv.writer(f, delimiter='\n')
    writer.writerow(labels)
    print('labels saved to file "labels" in project directory')

