import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, PReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import math

# ECA Attention Mechanisms
def eca_block(inputs, b=1, gamma=2):
    in_channel = inputs.shape[-1]
    kernel_size = int(abs((math.log(in_channel, 2) + b) / gamma))
    kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
    
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    x = tf.keras.layers.Reshape((in_channel, 1))(x)
    x = tf.keras.layers.Conv1D(filters=1, kernel_size=kernel_size, padding='same', use_bias=False)(x)
    x = tf.nn.sigmoid(x)
    x = tf.keras.layers.Reshape((1, 1, in_channel))(x)
    outputs = tf.keras.layers.multiply([inputs, x])
    return outputs

# Define the center crop function
def center_crop(img, target_height=224, target_width=224):
    height, width, _ = img.shape
    start_x = (width - target_width) // 2
    start_y = (height - target_height) // 2
    cropped_img = img[start_y:start_y+target_height, start_x:start_x+target_width, :]
    return cropped_img

base_dir = 'bird_data' 
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
target_size = (224, 224)
num_classes = 200
batch_size = 32

# Create an image data generator and perform data enhancement
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    preprocessing_function=center_crop
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=center_crop
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical'
)

base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(1024)(x)
x = PReLU()(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Iterate through each layer of the model and add ECA attention mechanisms
for layer in model.layers:
    if 'conv2d' in layer.name:
        layer_output = layer.output
        eca = eca_block(layer_output)
        model.get_layer(layer.name).output = eca

model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Adding Callback Functions
early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
model_checkpoint = ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=100,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size,
    callbacks=[early_stop, model_checkpoint, reduce_lr]
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()