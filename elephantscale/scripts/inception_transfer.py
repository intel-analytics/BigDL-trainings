
# Verify if the spark context was initialized 

#Import all the required packages

import numpy as np
import pandas as pd

from os import listdir
from os.path import join, basename
import struct
import json
from scipy import misc
import datetime as dt

from bigdl.nn.layer import *
from optparse import OptionParser
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *
from bigdl.dataset.transformer import *
from bigdl.nn.initialization_method import *
from transformer import *
from imagenet import *
from transformer import Resize

# if you want to train on whole imagenet
#from bigdl.dataset import imagenet

from pyspark import SparkConf, SparkContext

conf = SparkConf().setAppName("inception")
sc = SparkContext(conf=conf)

sc


# helper func to read the files from disk
def read_local_path(folder, has_label=True):
    """
    :param folder: local directory (str)
    :param has_label: does image have label (bool)
    :return: list of (image path , label) tuples
    """
    # read directory, create map
    dirs = listdir(folder)
    print "local path: ", folder
    print "listdir: ", dirs
    # create a list of (image path , label) tuples
    image_paths = []
    #append image path to the label (ex: )
    if has_label:
        dirs.sort()
        for d in dirs:
            for f in listdir(join(folder, d)):
                image_paths.append((join(join(folder, d), f), dirs.index(d) + 1))
    else:
        for f in dirs:
            image_paths.append((join(folder, f), -1))
    return image_paths

# helper func to read the files from disk
def read_local(sc, folder, normalize=255.0, has_label=True):
    """
    Read images from local directory
    :param sc: spark context
    :param folder: local directory
    :param normalize: normalization value
    :param has_label: whether the image folder contains label
    :return: RDD of sample
    """
    # read directory, create image paths list
    image_paths = read_local_path(folder, has_label)
    # print "BEFORE PARALLELIZATION: ", image_paths
    # create rdd
    image_paths_rdd = sc.parallelize(image_paths)
    # print image_paths_rdd
    feature_label_rdd = image_paths_rdd.map(lambda path_label: (misc.imread(path_label[0]), np.array(path_label[1]))) \
        .map(lambda img_label:
             (Resize(256, 256)(img_label[0]), img_label[1])) \
        .map(lambda feature_label:
             (((feature_label[0] & 0xff) / normalize).astype("float32"), feature_label[1]))
    return feature_label_rdd

def scala_T(input_T):
    """
    Helper function for building Inception layers. Transforms a list of numbers to a dictionary with ascending keys 
    and 0 appended to the front. Ignores dictionary inputs. 
    
    :param input_T: either list or dict
    :return: dictionary with ascending keys and 0 appended to front {0: 0, 1: realdata_1, 2: realdata_2, ...}
    """    
    if type(input_T) is list:
        # insert 0 into first index spot, such that the real data starts from index 1
        temp = [0]
        temp.extend(input_T)
        return dict(enumerate(temp))
    # if dictionary, return it back
    return input_T

# Question: What is config?
def Inception_Layer_v1(input_size, config, name_prefix=""):
    """
    Builds the inception-v1 submodule, a local network, that is stacked in the entire architecture when building
    the full model.  
    
    :param input_size: dimensions of input coming into the local network
    :param config: ?
    :param name_prefix: string naming the layers of the particular local network
    :return: concat container object with all of the Sequential layers' ouput concatenated depthwise
    """        
    
    '''
    Concat is a container who concatenates the output of it's submodules along the provided dimension: all submodules 
    take the same inputs, and their output is concatenated.
    '''
    concat = Concat(2)
    
    """
    In the above code, we first create a container Sequential. Then add the layers into the container one by one. The 
    order of the layers in the model is same with the insertion order. 
    
    """
    conv1 = Sequential()
    
    #Adding layes to the conv1 model we jus created
    
    #SpatialConvolution is a module that applies a 2D convolution over an input image.
    conv1.add(SpatialConvolution(input_size, config[1][1], 1, 1, 1, 1).set_name(name_prefix + "1x1"))
    conv1.add(ReLU(True).set_name(name_prefix + "relu_1x1"))
    concat.add(conv1)
    
    conv3 = Sequential()
    conv3.add(SpatialConvolution(input_size, config[2][1], 1, 1, 1, 1).set_name(name_prefix + "3x3_reduce"))
    conv3.add(ReLU(True).set_name(name_prefix + "relu_3x3_reduce"))
    conv3.add(SpatialConvolution(config[2][1], config[2][2], 3, 3, 1, 1, 1, 1).set_name(name_prefix + "3x3"))
    conv3.add(ReLU(True).set_name(name_prefix + "relu_3x3"))
    concat.add(conv3)
    
    
    conv5 = Sequential()
    conv5.add(SpatialConvolution(input_size,config[3][1], 1, 1, 1, 1).set_name(name_prefix + "5x5_reduce"))
    conv5.add(ReLU(True).set_name(name_prefix + "relu_5x5_reduce"))
    conv5.add(SpatialConvolution(config[3][1], config[3][2], 5, 5, 1, 1, 2, 2).set_name(name_prefix + "5x5"))
    conv5.add(ReLU(True).set_name(name_prefix + "relu_5x5"))
    concat.add(conv5)
    
    
    pool = Sequential()
    pool.add(SpatialMaxPooling(3, 3, 1, 1, 1, 1, to_ceil=True).set_name(name_prefix + "pool"))
    pool.add(SpatialConvolution(input_size, config[4][1], 1, 1, 1, 1).set_name(name_prefix + "pool_proj"))
    pool.add(ReLU(True).set_name(name_prefix + "relu_pool_proj"))
    concat.add(pool).set_name(name_prefix + "output")
    return concat

def Inception_v1_NoAuxClassifier(class_num):
    model = Sequential()
    model.add(SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3, 1, False).set_name("conv1/7x7_s2"))
    model.add(ReLU(True).set_name("conv1/relu_7x7"))
    model.add(SpatialMaxPooling(3, 3, 2, 2, to_ceil=True).set_name("pool1/3x3_s2"))
    model.add(SpatialCrossMapLRN(5, 0.0001, 0.75).set_name("pool1/norm1"))
    model.add(SpatialConvolution(64, 64, 1, 1, 1, 1).set_name("conv2/3x3_reduce"))
    model.add(ReLU(True).set_name("conv2/relu_3x3_reduce"))
    model.add(SpatialConvolution(64, 192, 3, 3, 1, 1, 1, 1).set_name("conv2/3x3"))
    model.add(ReLU(True).set_name("conv2/relu_3x3"))
    model.add(SpatialCrossMapLRN(5, 0.0001, 0.75).set_name("conv2/norm2"))
    model.add(SpatialMaxPooling(3, 3, 2, 2, to_ceil=True).set_name("pool2/3x3_s2"))
    model.add(Inception_Layer_v1(192, scala_T([scala_T([64]), scala_T(
         [96, 128]), scala_T([16, 32]), scala_T([32])]), "inception_3a/"))
    model.add(Inception_Layer_v1(256, scala_T([scala_T([128]), scala_T(
         [128, 192]), scala_T([32, 96]), scala_T([64])]), "inception_3b/"))
    model.add(SpatialMaxPooling(3, 3, 2, 2, to_ceil=True))
    model.add(Inception_Layer_v1(480, scala_T([scala_T([192]), scala_T(
         [96, 208]), scala_T([16, 48]), scala_T([64])]), "inception_4a/"))
    model.add(Inception_Layer_v1(512, scala_T([scala_T([160]), scala_T(
         [112, 224]), scala_T([24, 64]), scala_T([64])]), "inception_4b/"))
    model.add(Inception_Layer_v1(512, scala_T([scala_T([128]), scala_T(
         [128, 256]), scala_T([24, 64]), scala_T([64])]), "inception_4c/"))
    model.add(Inception_Layer_v1(512, scala_T([scala_T([112]), scala_T(
         [144, 288]), scala_T([32, 64]), scala_T([64])]), "inception_4d/"))
    model.add(Inception_Layer_v1(528, scala_T([scala_T([256]), scala_T(
         [160, 320]), scala_T([32, 128]), scala_T([128])]), "inception_4e/"))
    model.add(SpatialMaxPooling(3, 3, 2, 2, to_ceil=True))
    model.add(Inception_Layer_v1(832, scala_T([scala_T([256]), scala_T(
         [160, 320]), scala_T([32, 128]), scala_T([128])]), "inception_5a/"))
    model.add(Inception_Layer_v1(832, scala_T([scala_T([384]), scala_T(
         [192, 384]), scala_T([48, 128]), scala_T([128])]), "inception_5b/"))
    model.add(SpatialAveragePooling(7, 7, 1, 1).set_name("pool5/7x7_s1"))
    model.add(Dropout(0.4).set_name("pool5/drop_7x7_s1"))
    model.add(View([1024], num_input_dims=3))
    model.add(Linear(1024, class_num).set_name("loss3/classifier_flowers"))
    model.add(LogSoftMax().set_name("loss3/loss3"))
    model.reset()
    return model

def Inception_v1(class_num):
    """
    Builds the entire network using Inception architecture  
    
    :param class_num: number of categories of classification
    :return: entire model architecture 
    """
    #contains first 3 inception modules
    feature1 = Sequential()
    
    feature1.add(SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3, 1, False).set_name("conv1/7x7_s2"))
    feature1.add(ReLU(True).set_name("conv1/relu_7x7"))
    feature1.add(SpatialMaxPooling(3, 3, 2, 2, to_ceil=True).set_name("pool1/3x3_s2"))
    feature1.add(SpatialCrossMapLRN(5, 0.0001, 0.75).set_name("pool1/norm1"))
    feature1.add(SpatialConvolution(64, 64, 1, 1, 1, 1).set_name("conv2/3x3_reduce"))
    feature1.add(ReLU(True).set_name("conv2/relu_3x3_reduce"))
    feature1.add(SpatialConvolution(64, 192, 3, 3, 1, 1, 1, 1).set_name("conv2/3x3"))
    feature1.add(ReLU(True).set_name("conv2/relu_3x3"))
    feature1.add(SpatialCrossMapLRN(5, 0.0001, 0.75).set_name("conv2/norm2"))
    feature1.add(SpatialMaxPooling(3, 3, 2, 2, to_ceil=True).set_name("pool2/3x3_s2"))
    feature1.add(Inception_Layer_v1(192,scala_T([scala_T([64]), scala_T([96, 128]),scala_T([16, 32]), scala_T([32])]),
                                    "inception_3a/"))
    feature1.add(Inception_Layer_v1(256, scala_T([scala_T([128]), scala_T([128, 192]), scala_T([32, 96]), scala_T([64])]),
                                    "inception_3b/"))
    feature1.add(SpatialMaxPooling(3, 3, 2, 2, to_ceil=True).set_name("pool3/3x3_s2"))
    feature1.add(Inception_Layer_v1(480, scala_T([scala_T([192]), scala_T([96, 208]), scala_T([16, 48]), scala_T([64])]),
                                    "inception_4a/"))
    # 1st classification ouput after 3 inception subnetworks
    output1 = Sequential()
    output1.add(SpatialAveragePooling(5, 5, 3, 3, ceil_mode=True).set_name("loss1/ave_pool"))
    output1.add(SpatialConvolution(512, 128, 1, 1, 1, 1).set_name("loss1/conv"))
    output1.add(ReLU(True).set_name("loss1/relu_conv"))
    output1.add(View([128 * 4 * 4], num_input_dims = 3))
    output1.add(Linear(128 * 4 * 4, 1024).set_name("loss1/fc"))
    output1.add(ReLU(True).set_name("loss1/relu_fc"))
    output1.add(Dropout(0.7).set_name("loss1/drop_fc"))
    output1.add(Linear(1024, class_num).set_name("loss1/classifier_5classes"))
    output1.add(LogSoftMax().set_name("loss1/loss"))

    # contains next 3 inception submodules
    feature2 = Sequential()
    feature2.add(Inception_Layer_v1(512, scala_T([scala_T([160]), scala_T([112, 224]),scala_T([24, 64]), scala_T([64])]),
                                    "inception_4b/"))
    feature2.add(Inception_Layer_v1(512, scala_T([scala_T([128]), scala_T([128, 256]),scala_T([24, 64]), scala_T([64])]),
                                    "inception_4c/"))
    feature2.add(Inception_Layer_v1(512, scala_T([scala_T([112]), scala_T([144, 288]), scala_T([32, 64]), scala_T([64])]),
                                    "inception_4d/"))
    # 2nd classification output after 3 more inception subnetworks
    output2 = Sequential()
    output2.add(SpatialAveragePooling(5, 5, 3, 3).set_name("loss2/ave_pool"))
    output2.add(SpatialConvolution(528, 128, 1, 1, 1, 1).set_name("loss2/conv"))
    output2.add(ReLU(True).set_name("loss2/relu_conv"))
    output2.add(View([128 * 4 * 4], num_input_dims=3))
    output2.add(Linear(128 * 4 * 4, 1024).set_name("loss2/fc"))
    output2.add(ReLU(True).set_name("loss2/relu_fc"))
    output2.add(Dropout(0.7).set_name("loss2/drop_fc"))
    output2.add(Linear(1024, class_num).set_name("loss2/classifier_5classes"))
    output2.add(LogSoftMax().set_name("loss2/loss"))

    # final 3 inception submodules followed by linear/softmax
    output3 = Sequential()
    output3.add(Inception_Layer_v1(528, scala_T([scala_T([256]), scala_T([160, 320]), scala_T([32, 128]), scala_T([128])]),
                                   "inception_4e/"))
    output3.add(SpatialMaxPooling(3, 3, 2, 2, to_ceil=True).set_name("pool4/3x3_s2"))
    output3.add(Inception_Layer_v1(832, scala_T([scala_T([256]), scala_T([160, 320]), scala_T([32, 128]), scala_T([128])]),
                                   "inception_5a/"))
    output3.add(Inception_Layer_v1(832,scala_T([scala_T([384]), scala_T([192, 384]),scala_T([48, 128]), scala_T([128])]),
                                   "inception_5b/"))
    output3.add(SpatialAveragePooling(7, 7, 1, 1).set_name("pool5/7x7_s1"))
    output3.add(Dropout(0.4).set_name("pool5/drop_7x7_s1"))
    output3.add(View([1024], num_input_dims=3))
    output3.add(Linear(1024, class_num).set_name("loss3/classifier_5classes"))
    output3.add(LogSoftMax().set_name("loss3/loss3"))

    # Attach the separate Sequential layers to create the whole model
    split2 = Concat(2).set_name("split2")
    split2.add(output3)
    split2.add(output2)

    #create a branch starting from feature2 upwards
    mainBranch = Sequential()
    mainBranch.add(feature2)
    mainBranch.add(split2)

    #concatenate the mainBranch with output1
    split1 = Concat(2).set_name("split1")
    split1.add(mainBranch)
    split1.add(output1)

    #Attach feature1 to the rest of the model
    model = Sequential()

    model.add(feature1)
    model.add(split1)

    model.reset()
    return model

def get_inception_data(folder, file_type="image", data_type="train", normalize=255.0):
    """
    Builds the entire network using Inception architecture  
    
    :param class_num: number of categories of classification
    :return: entire model architecture 
    """
    #Getting the path of our data
    path = os.path.join(folder, data_type)
    if "seq" == file_type:
        return read_seq_file(sc, path, normalize)
    elif "image" == file_type:
        return read_local(sc, path, normalize)

# initializing BigDL engine
init_engine()

# paths for datasets, saving checkpoints 

DATA_PATH = "./sample_images/"
checkpoint_path = "./sample_images/checkpoints"

#providing the no of classes in the dataset to model (5 for flowers)
classNum = 5

# Instantiating the model the model
inception_model = Inception_v1_NoAuxClassifier(classNum)

# path, names of the downlaoded pre-trained caffe models
caffe_prototxt = 'bvlc_googlenet.prototxt'
caffe_model = 'bvlc_googlenet.caffemodel'

# loading the weights to the BigDL inception model, EXCEPT the weights for the last fc layer (classification layer)
model = Model.load_caffe(inception_model, caffe_prototxt, caffe_model, match_all=False, bigdl_type="float")

# if we want to export the whole caffe model including definition, this can be used.
#model = Model.load_caffe_model(inception_model, caffe_prototxt, caffe_model, match_all=True)

# Get the flower categories
from os import listdir
from os.path import isfile, join
labels = listdir("./sample_images/flower_photos")
print "labels: ", labels

'''
Helper function to make sure image width or height is no smaller than 224x224
'''
def adjust_dimensions(input_img):
    
    dimensions = input_img.getbbox()
    if dimensions[2] < 224:
        input_img = input_img.resize((224,dimensions[3]))
        dimensions = input_img.getbbox()
        # print "1", dimensions
    if dimensions[3] < 224:
        input_img = input_img.resize((dimensions[2],224))
        # print "2", input_img.getbbox()
    return input_img

'''
Convert the test image to an Image object
Note: Images in vegnonveg-sample are large and maybe need to be cropped/resized before being trained on.
'''
from PIL import Image # for seeing image
import cv2 # converting img to numpy array (RGB to BGR) 

sample_images_path = "./sample_images/flower_photos/sunflowers/"
input_str = '1008566138_6927679c8a.jpg'
input_img = Image.open(sample_images_path + input_str)
input_img = adjust_dimensions(input_img)
print "accepted image dimensions: ", input_img.getbbox()
# DISPLAY IMAGE HERE
input_img.show()

'''
Convert Image object into a 3d numpy array representing BGR of each pixel (for bigdl)
'''
# img = cv2.imread(sample_images_path + input_str)  
# # print img
img = np.array(input_img)
img = img[:,:,::-1].copy() #invert RGB representation to BGR for each pixel in img
img.shape

'''
Normalize, crop, finish pre-processing of image so it can be fed to rdd[sample] for bigdl.
'''
# defining the transformer, which we will use to pre-process our test image
img_rows = 224
img_cols = 224


transform_input = Transformer([Crop(img_rows, img_cols, "center"),
                                        ChannelNormalizer(0.485, 0.456, 0.406, 0.229, 0.224, 0.225),
                                        TransposeToTensor(False)
                                        ])
# pre-processing the img, feature transformation decreases training time
img_tranx = transform_input(img)

'''
Converting the image to 'Sample' format which BigDL expects. 
'''
label = np.array(1) #label of dandelion
img_to_model = Sample.from_ndarray(img_tranx, label)

'''
Converting image from 'Sample' format into RDD format
'''
img_data_rdd = sc.parallelize([img_to_model])

# predicting the image using our model
predict_result = model.predict_class(img_data_rdd)
pred_index = predict_result.collect()[0]
print pred_index

# printing out the category 
if pred_index > classNum - 1 :
    pred_index = pred_index % classNum
    
class_predicted = str(labels[pred_index - 1])
print (class_predicted)

'''
GOAL: Predicts the first 20 images in a specified flower folder using the pre-trained model
'''
from PIL import Image # for seeing image
import cv2 # converting img to numpy array (RGB to BGR) 

# get local path of each image
flower = "sunflowers"
dand_path = "./sample_images/flower_photos/" + flower + "/"
imgs = listdir(dand_path)

# predict first 20 images
for img in imgs[0:19]:
    input_img = Image.open(dand_path + img)
    input_img = adjust_dimensions(input_img)
    
    # convert img to np.array form
    img_bgr = np.array(input_img)
    img_bgr = img_bgr[:,:,::-1].copy()
    img_tranx = transform_input(img_bgr)   
    
    #get label of flower
    label = np.array(labels.index(flower)) 
    
    # converting to 'Sample' format which BigDL expects. 
    img_to_model = Sample.from_ndarray(img_tranx, label)    
    
    # converting from 'Sample' format into RDD format
    img_data_rdd = sc.parallelize([img_to_model])
    
    # predicting the image using our model
    predict_result = model.predict_class(img_data_rdd)
    pred_index = predict_result.collect()[0]   

    # printing out the category 
    if pred_index > classNum:
        print pred_index
        pred_index = pred_index % classNum
    class_predicted = str(labels[pred_index - 1])
    print (class_predicted)
    


# the image size expected by the model
image_size = 224

# image transformer, used for pre-processing the train images 
train_transformer = Transformer([Crop(image_size, image_size),
                                  Flip(0.5),
                                  ChannelNormalizer(0.485, 0.456, 0.406, 0.229, 0.224, 0.225),
                                  TransposeToTensor(False)])

# reading the traning 

'''
Goal: Predict first 20 images in the "dandelion" folder using un-trained model.
ERROR: py4j.Py4JException: Method modelPredictClass([class com.intel.analytics.bigdl.nn.Sequential, class java.util.ArrayList]) does not exist
'''
from PIL import Image # for seeing image
import cv2 # converting img to numpy array (RGB to BGR) 

def map_groundtruth_label(l):
    return l.to_ndarray()[0] - 1
# defining the tranformer, which we will use to pre-process our test image

img_rows = 224
img_cols = 224


transform_input = Transformer([Crop(img_rows, img_cols, "center"),
                                        Flip(0.5),
                                        ChannelNormalizer(0.485, 0.456, 0.406, 0.229, 0.224, 0.225),
                                        TransposeToTensor(False)
                                        ])
dand_path = "./sample_images/flower_photos/dandelion/"
imgs = listdir(dand_path)

#get paths
img_paths = []
for img in imgs[0:20]:
    img_paths.append((dand_path+img, 1))

print ("img_paths:")
print (img_paths)
    
#turn img_paths into labelled rdds
image_paths_rdd = sc.parallelize(img_paths)

feature_label_rdd = image_paths_rdd.map(lambda path_label: (misc.imread(path_label[0]), np.array(path_label[1]))) \
        .map(lambda img_label:
             (Resize(256, 256)(img_label[0]), img_label[1])) \
        .map(lambda feature_label:
             (((feature_label[0] & 0xff) / 255.0).astype("float32"), feature_label[1]))

print("Feature label:")
print(feature_label_rdd.take(10))

#turn to sample form for predictions 
img_data = feature_label_rdd.map(
                lambda features_label: (train_transformer(features_label[0]), features_label[1])).map(
                lambda features_label: Sample.from_ndarray(features_label[0], features_label[1] + 1))

# predicting the image using our model
print "Predictions: "
res = model.predict_class(img_data)
print res.collect()

print "True Labels: "
print ', '.join(str(map_groundtruth_label(s.label)) for s in img_data.take(10))


'''
Goal: Predict first 20 images in the "dandelion" folder using un-trained model.
ERROR: py4j.Py4JException: Method modelPredictClass([class com.intel.analytics.bigdl.nn.Sequential, class java.util.ArrayList]) does not exist
'''
from PIL import Image # for seeing image
import cv2 # converting img to numpy array (RGB to BGR) 

# defining the tranformer, which we will use to pre-process our test image

img_rows = 224
img_cols = 224


transform_input = Transformer([Crop(img_rows, img_cols, "center"),
                                        Flip(0.5),
                                        ChannelNormalizer(0.485, 0.456, 0.406, 0.229, 0.224, 0.225),
                                        TransposeToTensor(False)
                                        ])
dand_path = "./sample_images/flower_photos/dandelion/"
imgs = listdir(dand_path)




#get paths
img_paths = []
for img in imgs[0:20]:
    img_paths.append((dand_path+img, 1))
    

img_paths

'''
Reading the training and validation data and perform pre-processing 
'''


# the image size expected by the model
image_size = 224

# image transformer, used for pre-processing the train images 
train_transformer = Transformer([Crop(image_size, image_size),
                                  Flip(0.5),
                                  ChannelNormalizer(0.485, 0.456, 0.406, 0.229, 0.224, 0.225),
                                  TransposeToTensor(False)])

# reading the traning data
train_data = get_inception_data(DATA_PATH, "image", "train").map(
                lambda features_label: (train_transformer(features_label[0]), features_label[1])).map(
                lambda features_label: Sample.from_ndarray(features_label[0], features_label[1] + 1))

# validation data transformer 
val_transformer = Transformer([Crop(image_size, image_size, "center"),
                                Flip(0.5),
                                ChannelNormalizer(0.485, 0.456, 0.406, 0.229, 0.224, 0.225),
                                TransposeToTensor(False)])

#reading the validation data
val_data = get_inception_data(DATA_PATH, "image", "val").map(
                lambda features_label: (val_transformer(features_label[0]), features_label[1])).map(
                lambda features_label: Sample.from_ndarray(features_label[0], features_label[1] + 1))



# training the model
# parameters for 
batch_size = 16
no_epochs = 2

# Optimizer
optimizer = Optimizer(
                model=model,
                training_rdd=train_data.filter(lambda l: l.label.to_ndarray()[0] <= 5),
                #optim_method=Adam(learningrate=0.002),
                optim_method = SGD(learningrate=0.01, learningrate_decay=0.0002),
                criterion=ClassNLLCriterion(),
                end_trigger=MaxEpoch(no_epochs),
                batch_size=batch_size
            )

# setting checkpoints
optimizer.set_checkpoint(EveryEpoch(), checkpoint_path, isOverWrite=False)

# setting validation parameters 
optimizer.set_validation( batch_size=batch_size,
                          val_rdd=val_data,
                          trigger=EveryEpoch(),
                          val_method=[Top1Accuracy()])



# Log the training process to measure loss/accuracy, can be 
app_name= 'inception-' + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
train_summary = TrainSummary(log_dir='/tmp/inception_summaries',
                                     app_name=app_name)
train_summary.set_summary_trigger("Parameters", SeveralIteration(50))
val_summary = ValidationSummary(log_dir='/tmp/inception_summaries',
                                        app_name=app_name)
optimizer.set_train_summary(train_summary)
optimizer.set_val_summary(val_summary)
print "saving logs to ",app_name


# Boot training process
trained_model = optimizer.optimize()
print "Optimization Done."


# image transformer, used for pre-processing the validation images 
test_transformer = Transformer([Crop(image_size, image_size, "center"),
                                ChannelNormalizer(0.485, 0.456, 0.406, 0.229, 0.224, 0.225),
                                TransposeToTensor(False)])

# reading val data 
# get_inception_data() returns a PythonRDD
test_data = get_inception_data(DATA_PATH, "image", "test").map(
                lambda features_label: (test_transformer(features_label[0]), features_label[1])).map(
                lambda features_label: Sample.from_ndarray(features_label[0], features_label[1] + 1))


print "Predictions: "
res = trained_model.predict_class(test_data)
print res.collect()

print "True Labels: "
print ', '.join(str(map_groundtruth_label(s.label)) for s in test_data.take(8))
# testing the trained model 
results = trained_model.evaluate(test_data, batch_size, [Top1Accuracy()])

# Output results
for i in results:
  print (i)



