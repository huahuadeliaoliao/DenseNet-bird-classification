import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Conv2D, Multiply, Reshape, PReLU, GlobalMaxPooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
from sklearn.metrics import classification_report

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
batch_size = 8

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
    brightness_range=[0.8, 1.2],
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

# Constructing Feature Channel Attention Blocks (SE Blocks) and Using PReLUs
def se_block(input_tensor, ratio=16):
    init = input_tensor
    channel_axis = -1 if tf.keras.backend.image_data_format() == 'channels_last' else 1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, use_bias=False, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(se)
    se = PReLU()(se)
    se = Dense(filters, use_bias=False, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(se)
    se = PReLU()(se)

    return Multiply()([init, se])

# Modify DenseNet121 to add SE blocks
def modified_densenet121(input_shape=(224, 224, 3), num_classes=200):
    base_model = DenseNet121(include_top=False, weights='imagenet', input_tensor=None, input_shape=input_shape)

    x = base_model.output
    for i, layer in enumerate(base_model.layers):
        if isinstance(layer, tf.keras.layers.Activation):
            if 'relu' in layer.get_config()['activation']:
                x = se_block(layer.output)

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(1024, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
    x = PReLU()(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=x)
    return model

model = modified_densenet121(input_shape=(224, 224, 3), num_classes=num_classes)

optimizer = Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Adding Callback Functions
early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
model_checkpoint = ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=300,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size,
    callbacks=[early_stop, model_checkpoint, reduce_lr]
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()