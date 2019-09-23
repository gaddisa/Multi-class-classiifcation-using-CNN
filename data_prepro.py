"""
###################################################################################################
    Last updated on Sun May 12 15:44:14 2019
    This code contains all the answer to question 1 to 4, and it's called inside the
    main function
    
    Structure of my CNN Network
        Input Layer
        first conv layer(with relu)
        pooling layer
        second conv layer(with relu)
        pooling layer
        Flatten layer
        fully connected layer1
        output layer(with softmax)
    AdamOptimizer with cross entropy loss is used for learning
    
    @author: gaddisa olani
###################################################################################################
"""
import os
import glob
import numpy as np
import cv2
import random
from sklearn.utils import shuffle

"""
####################################################################################################
                Data Preprocessing (Augumentation)
            The name of the function indicates what they do on the original image
            The preprocssing is out is saved for each image in the form of multidimensional array
            All the following operations are applied to each image one by one.
#####################################################################################################
                
"""


def scale_image(image):
    fx=random.uniform(0, 3)
    fy=fx
    image = cv2.resize(image,None,fx=fx, fy=fy, interpolation = cv2.INTER_CUBIC)
    return image
    
def add_light(image):
    gamma=random.randint(1,5)
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    image=cv2.LUT(image, table)
    return image

def translation_image(image):
    x=random.randint(100,200)
    y=random.randint(100,200)
    rows, cols ,c= image.shape
    M = np.float32([[1, 0, x], [0, 1, y]])
    image = cv2.warpAffine(image, M, (cols, rows))
    return image

def rotate(img):
    #randomly rotate the image #
    angle=random.randint(2,360)
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(img, M, (w, h))
    return rotated_image
    

def saturate_image(image):
    saturation=random.randint(50,200)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    v = image[:, :, 2]
    v = np.where(v <= 255 - saturation, v + saturation, 255)
    image[:, :, 2] = v
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return image

def gausian_blur(image):
    blur=random.uniform(0, 1)
    image = cv2.GaussianBlur(image,(5,5),blur)
    return image

#this oepration involves dialation,erosion and substraction
def morphological_gradient_image(image):
    shift=random.randint(2,30)
    kernel = np.ones((shift, shift), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    return image
#to add a random saturation jitter to an image. 

def addeptive_gaussian_noise(image):
    h,s,v=cv2.split(image)
    s = cv2.adaptiveThreshold(s, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    h = cv2.adaptiveThreshold(h, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    v = cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    image=cv2.merge([h,s,v])
    return image

def contrast_image(image):
    contrast=random.randint(10,100)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image[:,:,2] = [[max(pixel - contrast, 0) if pixel < 190 else min(pixel + contrast, 255) for pixel in row] for row in image[:,:,2]]
    image= cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return image

def original(x):
    return x

def sharpen_image(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image = cv2.filter2D(image, -1, kernel)
    return image

def edge_image(image,ksize):
    image = cv2.Sobel(image,cv2.CV_16U,1,0,ksize=ksize)
    return image

def flip(x):
    x=cv2.flip(x,1)
    return x

def shift(x):
    """shift the image location"""
    
    rows,cols,c = x.shape
    M = np.float32([[1,0,100],[0,1,50]])
    x = cv2.warpAffine(x,M,(cols,rows))
    return x

    
def load_train(train_path, image_size, classes):
    images = []
    labels = []
    ids = []
    cls = []

    print('Reading training images,please wait')
    
    """
    #######################################################################################
    collect files from each folder  
    The folder label indicates the class label
    Thus each file in the same folder are labeled by the same class label (folder name)
    #######################################################################################        
        """
    
    
    for fld in classes:   
        index = classes.index(fld)
        path = os.path.join(train_path, fld, '*.jpg')
        
        files = glob.glob(path)
        
        count_number_samples=0
        
        """
        to make the function call easy I put all the function I defined earlier to this list,
        and call them one by one (one after another for each image)
        """
        augmentations = [original,translation_image,add_light,rotate,saturate_image,gausian_blur,morphological_gradient_image,
                         addeptive_gaussian_noise,sharpen_image]
        
        for fl in files:
            image_original = cv2.imread(fl)
            for f in augmentations:
                #while(count_number_samples<=26):
                image=f(image_original)
                """
                Rescale the image so that they have the same dimension
                and also perform Linear interpolation to fill missing value
                """
                image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
                images.append(image)
                label = np.zeros(len(classes))
                label[index] = 1.0
                labels.append(label)
                flbase = os.path.basename(fl)
                ids.append(flbase)
                cls.append(fld)
            count_number_samples+=1
            
                          
    images = np.array(images)
    labels = np.array(labels)
    ids = np.array(ids)
    cls = np.array(cls)
    
    return images, labels, ids, cls


def load_test(test_path, image_size,classes):

    images = []
    labels = []
    ids = []
    cls = []

    print('Reading test images,please wait')
    for fld in classes:   
        """
         apply the same logic to collect and and it's corresponding class label
        """
        index = classes.index(fld)
        path = os.path.join(test_path, fld, '*.jpg')
        
        files = glob.glob(path)
        
        
        for fl in files:
            image = cv2.imread(fl)
            image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
            images.append(image)
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)
            flbase = os.path.basename(fl)
            ids.append(flbase)
            cls.append(fld)
    images = np.array(images)
    labels = np.array(labels)
    ids = np.array(ids)
    cls = np.array(cls)
    
    return images, labels, ids, cls



class DataSet(object):

  def __init__(self, images, labels, ids, cls):
    """Construct a DataSet. one_hot arg is used only if fake_data is true."""

    self._num_examples = images.shape[0]


    # Convert shape from [num examples, rows, columns, depth]
    # to [num examples, rows*columns] (assuming depth == 1)
    # Convert from [0, 255] -> [0.0, 1.0].

    images = images.astype(np.float32)
    """
    Image Normalization begins here"""
    images = np.multiply(images, 1.0 / 255.0)

    self._images = images
    self._labels = labels
    self._ids = ids
    self._cls = cls
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def ids(self):
    return self._ids

  @property
  def cls(self):
    return self._cls

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size

    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch

    return self._images[start:end], self._labels[start:end], self._ids[start:end], self._cls[start:end]


def read_train_sets(train_path, image_size, classes, validation_size=0):
  class DataSets(object):
    pass
  data_sets = DataSets()

  images, labels, ids, cls = load_train(train_path, image_size, classes)
  images, labels, ids, cls = shuffle(images, labels, ids, cls)  # shuffle the data

  if isinstance(validation_size, float):
    validation_size = int(validation_size * images.shape[0])

  validation_images = images[:validation_size]
  validation_labels = labels[:validation_size]
  validation_ids = ids[:validation_size]
  validation_cls = cls[:validation_size]

  train_images = images[validation_size:]
  train_labels = labels[validation_size:]
  train_ids = ids[validation_size:]
  train_cls = cls[validation_size:]

  data_sets.train = DataSet(train_images, train_labels, train_ids, train_cls)
  data_sets.valid = DataSet(validation_images, validation_labels, validation_ids, validation_cls)

  return data_sets


def read_test_set(test_path, image_size,classes):
  class DataSets(object):
    pass
  test_sets = DataSets()
  images, labels, ids, cls  = load_test(test_path, image_size,classes)
  
  images, labels, ids, cls=shuffle(images, labels, ids, cls)
  
  test_sets=DataSet(images, labels, ids, cls)
  return test_sets
