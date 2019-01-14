# Transfer-Learning
Transfer learning using VGG16 on Fish dataset from Kaggle. link: https://www.kaggle.com/narae78/fish-detection/data.

# Prerequisite:
- Tensorflow 1.6+
- Keras
- CV2

# Custom Data: 
If you intend to use the above code on your own data, update the data folder. The structure of the folder should be such that the sub-folders are the class labels each containing the images belonging to that class.
Example: Data( contains following subfolders)
            -Dog ( Contains pictures with ground truth label "Dog")
            -Cat ( Contains pictures with ground truth label "Cat")
            -Bird ( Contains pictures with ground truth label "Bird")

# Download Weights: 
download vgg16.npy file from https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM.

# To-Run: 
Once you have checked paths to your data directory in tranferModel.py run it. It will internally call createTransferCodes.py which will create a label file and a codes file. 
The labels file will contain the ground truth labels of your data while the codes file will store the activations from relu6 layer of the VGG16 model.

To change the layer from which you want your activations from update the createTranferCodes.py.
Line: codes_batch = sess.run(vgg.relu6, feed_dict=feed_dict) in creatTransferCodes.py.
To see the names of each layer in vgg16 model see vgg16.py.
