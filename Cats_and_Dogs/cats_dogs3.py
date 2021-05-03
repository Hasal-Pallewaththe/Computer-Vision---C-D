# This is the Third CNN - results are given in the "results" directory

import os
import random
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2


# please visit this site: https://www.microsoft.com/en-us/download/confirmation.aspx?id=54765
# Download the dataset for Cats and Dogs, extract the downloaded folder and paste the PetImages folder here


print('number of Total Cat images = ' + str(len(os.listdir('PetImages/Cat/'))))
print('number of Total Dog images = ' + str(len(os.listdir('PetImages/Dog/'))))

# there are 12501 dogs and 12501 cat images in PetImages directory, output should be,
# number of Total Cat images =12501
# number of Total Dog images =12501

try:
    os.mkdir('data')
    os.mkdir('data/training')
    os.mkdir('data/training/cats')
    os.mkdir('data/training/dogs')
    os.mkdir('data/testing')
    os.mkdir('data/testing/cats')
    os.mkdir('data/testing/dogs')
except OSError:
    pass


def split_data(PetImage_Dir, Training_Dir, Testing_Dir, Size):
    files = []
    for filename in os.listdir(PetImage_Dir):
        file = PetImage_Dir + filename
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " filesize is empty,file is corrupted, therefore removed")

    training_length = int(len(files) * Size)
    testing_length = int(len(files) - training_length)
    shuffled_set = random.sample(files, len(files))
    training_set = shuffled_set[0:training_length]
    testing_set = shuffled_set[-testing_length:]

    for filename in training_set:
        this_file = PetImage_Dir + filename
        destination = Training_Dir + filename
        copyfile(this_file, destination)

    for filename in testing_set:
        this_file = PetImage_Dir + filename
        destination = Testing_Dir + filename
        copyfile(this_file, destination)


Cat_PetImage_Dir = "PetImages/Cat/"
Cat_Training_Dir = "data/training/cats/"
Cat_Testing_Dir = "data/testing/cats/"
Dog_PetImage_Dir = "PetImages/Dog/"
Dog_Training_Dir = "data/training/dogs/"
Dog_Testing_Dir = "data/testing/dogs/"

split_size = 0.9
split_data(Cat_PetImage_Dir, Cat_Training_Dir, Cat_Testing_Dir, split_size)
split_data(Dog_PetImage_Dir, Dog_Training_Dir, Dog_Testing_Dir, split_size)

# Expected output
# 666.jpg is zero length, so ignoring
# 11702.jpg is zero length, so ignoring

print('number of Training Cat images = ' + str(len(os.listdir('data/training/cats/'))))
print('number of Training Dog images = ' + str(len(os.listdir('data//training/dogs/'))))
print('number of Testing Cat images = ' + str(len(os.listdir('data/testing/cats/'))))
print('number of Testing Dog images = ' + str(len(os.listdir('data/testing/dogs/'))))


# output should be,
# number of Training Cat images = 11250
# number of Training Dog images = 11250
# number of Testing Cat images = 1250
# number of Testing Dog images = 1250


# adding an early stop to reduce overfitting
callback_end = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
    # tf.keras.layers.MaxPooling2D(2, 2),
    # tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
    # tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.3),  # use this dropout class to reduce overfittig
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])


Training_Path = "data/training/"
Training_imgdatagen = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=40,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    fill_mode='nearest'
                                        )

Training_gen = Training_imgdatagen.flow_from_directory(
                            Training_Path,
                            batch_size=50,
                            class_mode='binary',
                            target_size=(150, 150)  # All images are resized to 150x150
                                                   )

Teating_Path = "data/testing/"
Validation_imgdatagen = ImageDataGenerator(rescale=1./255)

Validation_gen = Validation_imgdatagen.flow_from_directory(
                                        Teating_Path,
                                        batch_size=50,
                                        class_mode='binary',
                                        target_size=(150, 150)  # All images are resized to 150x150
                                                              )


history = model.fit(Training_gen, epochs=100, steps_per_epoch=450,
                    validation_data=Validation_gen, validation_steps=50, callbacks=[callback_end])


# Obtain the accuracy and error result values as a list to be used in graphs


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, (len(acc)+1))  # Get number of epochs


# Plotting training and validation accuracy vs epochs graph

plt.figure(1)
plt.plot(epochs, acc, 'r', label="Training Accuracy")
plt.plot(epochs, val_acc, 'b', label="Validation Accuracy")
# plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
plt.legend()
plt.title('Training and validation Accuracy')
plt.xlabel('number of epochs')


# Plotting training and validation loss (error) vs epochs graph

plt.figure(2)
plt.plot(epochs, loss, 'r', label="Training Error")
plt.plot(epochs, val_loss, 'b', label="Validation Error")
# plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
plt.legend()
plt.title('Training and validation Error(Loss)')
plt.xlabel('number of epochs')

plt.show()
