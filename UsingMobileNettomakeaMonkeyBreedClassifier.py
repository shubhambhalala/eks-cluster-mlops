#!/usr/bin/env python
# coding: utf-8

# # Using MobileNet for our Monkey Classifer
# 
# ### Loading the MobileNet Model

# Freeze all layers except the top 4, as we'll only be training the top 4

from keras.applications import MobileNet

# MobileNet was designed to work on 224 x 224 pixel input images sizes
img_rows, img_cols = 224, 224 

# Re-loads the MobileNet model without the top or FC layers
MobileNet = MobileNet(weights = 'imagenet', include_top = False, input_shape = (img_rows, img_cols, 3))

# Here we freeze the last 4 layers 
# Layers are set to trainable as True by default
for layer in MobileNet.layers:
    layer.trainable = False
    
# Let's print our layers 
for (i,layer) in enumerate(MobileNet.layers):
    print(str(i) + " "+ layer.__class__.__name__, layer.trainable)


# ### Let's make a function that returns our FC Head

def addlayer(bottom_model, num_classes):
    """creates the top or head of the model that will be 
    placed ontop of the bottom layers"""

    top_model = bottom_model.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dense(1024,activation='relu')(top_model)
    top_model = Dense(num_classes,activation='softmax')(top_model)
    return top_model


# ### Let's add our FC Head back onto MobileNet



from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model

# Set our class number to 3 (Young, Middle, Old)
num_classes = 10

FC_Head = addlayer(MobileNet, num_classes)

model = Model(inputs = MobileNet.input, outputs = FC_Head)

print(model.summary())


# ### Loading our Monkey Breed Dataset

from keras.preprocessing.image import ImageDataGenerator

train_data_dir = '/root/monkey_breed/monkey_breed/train/'
validation_data_dir = '/root/monkey_breed/monkey_breed/validation/'

# Let's use some data augmentaiton 
train_datagen = ImageDataGenerator( rescale=1./255, rotation_range=45, width_shift_range=0.3, height_shift_range=0.3, horizontal_flip=True, fill_mode='nearest')
 
validation_datagen = ImageDataGenerator(rescale=1./255)
 
# set our batch size (typically on most mid tier systems we'll use 16-32)
batch_size = 32
 
train_generator = train_datagen.flow_from_directory( train_data_dir, target_size=(img_rows, img_cols), batch_size=batch_size, class_mode='categorical')
 
validation_generator = validation_datagen.flow_from_directory( validation_data_dir, target_size=(img_rows, img_cols), batch_size=batch_size, class_mode='categorical')


# ### Training out Model
# - Note we're using checkpointing and early stopping

from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping

                     
checkpoint = ModelCheckpoint("monkey_breed_mobileNet.h5",monitor="val_loss",mode="min",save_best_only = True,verbose=1)

earlystop = EarlyStopping(monitor = 'val_loss',min_delta = 0,patience = 3,verbose = 1,restore_best_weights = True)

# we put our call backs into a callback list
callbacks = [earlystop, checkpoint]

# We use a very small learning rate 
model.compile(loss = 'categorical_crossentropy',optimizer = RMSprop(lr = 0.001),metrics = ['accuracy'])

# Enter the number of training and validation samples here
nb_train_samples = 1097
nb_validation_samples = 272

# We only train 5 EPOCHS 
epochs = 5
batch_size = 16

history = model.fit_generator(train_generator,steps_per_epoch = nb_train_samples // batch_size,epochs = epochs,callbacks = callbacks,validation_data= validation_generator,validation_steps = nb_validation_samples // batch_size)

final_accuracy=history.history["val_accuracy"][-1]
print(final_accuracy)

import os
"""if final_accuracy < 0.92:
    os.system("curl --user '<jenkins username>:<jenkins_password>' http://192.168.99.102:8080/view/eks-mlops/job/eks-retrain/build?token=retrain")
else:
    print("Your New accuracy=",final_accuracy)"""
os.system("echo {0} > accuracy.txt".format(final_accuracy))
