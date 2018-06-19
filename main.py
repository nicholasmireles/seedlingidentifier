import tensorflow
import sys
import numpy as np
from keras.applications.xception import Xception, preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from keras.regularizers import l2
from keras.optimizers import RMSprop

# Some flag definitions
FINE_TUNE = True
CALCULATE_BN = True

# File related variables
train_dir = 'data/train/'
test_dir = 'data/test/'
bot_dir = 'data/bottleneck/'
model_name = 'Inception'

# Defining some parameters for the model
batch_size = 128
epochs = 12
dense1 = 32
dense2 = 12
pooling = 'avg'
num_freeze_layers = 249

''' 
 Number of layers to freeze:
 Xception: 126
 Inception v3: 249 <- This is the top 2 inception modules
'''

# Creating the generators for the images
train_gen = image.ImageDataGenerator(validation_split=.2, preprocessing_function=preprocess_input)
test_gen = image.ImageDataGenerator(preprocessing_function=preprocess_input)

# Creating the actual data generator objects for the different sets
train_data = train_gen.flow_from_directory(train_dir, subset="training", batch_size=batch_size, shuffle=False)
val_data = train_gen.flow_from_directory(train_dir, subset="validation", batch_size=batch_size, shuffle=False)
test_data = test_gen.flow_from_directory(test_dir, batch_size=batch_size, class_mode=None)

# Loading the Xception base model
print('Loading base model...')
# base_model = Xception(weights='imagenet',include_top=False,pooling=pooling)
base_model = InceptionV3(weights='imagenet', include_top=False, pooling=pooling)

if CALCULATE_BN:
    print('Calculating bottleneck features for dataset.')

    train_features = np.zeros((0, 2048))
    train_labels = np.zeros((0, 12))
    val_features = np.zeros((0, 2048))
    val_labels = np.zeros((0, 12))

    num_batches = len(train_data)
    i=1
    for batch_x, batch_y in train_data:
        print('Working on training batch: {}/{}'.format(i, num_batches))
        train_predictions = base_model.predict(batch_x)
        np.concatenate((train_features, train_predictions), axis=0)
        np.concatenate((train_labels, batch_y), axis=0)
        i+=1

    print('Saving training features to disk.')
    np.save(bot_dir + model_name + '_trainscores.npy', train_features)
    np.save(bot_dir + model_name + '_trainlabels.npy', train_labels)

    num_batches = len(val_data)
    i=1
    for batch_x, batch_y in val_data:
        print('Working on validation batch: {}/{}'.format(i, num_batches))
        val_predictions = base_model.predict(batch_x)
        np.concatenate((val_features, val_predictions), axis=0)
        np.concatenate((val_labels, batch_y), axis=0)
        i+=1

    print('Saving validation features to disk.')
    np.save(bot_dir + model_name + '_valscores.npy', val_features)
    np.save(bot_dir + model_name + '_vallabels.npy', val_labels)

else:
    print('Loading bottleneck scores from disk.')
    X_train = np.load(bot_dir + model_name + '_trainscores.npy')
    y_train = np.load(bot_dir + model_name + '_trainlabels.npy')
    X_val = np.load(bot_dir + model_name + '_valscores.npy')
    y_val = np.load(bot_dir + model_name + '_vallabels.npy')

# Freezing weights for transfer-learning:
for layer in base_model.layers:
    layer.trainable = False

# Adding in the new layers
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
# model.add(Dense(dense1,activation="relu"))
# model.add(Dropout(.5))
model.add(Dense(dense2, activation="softmax", kernel_regularizer=l2(0.01)))

# Compiling/fitting the model for transfer-learning
model.compile(optimizer="Adadelta", loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())
model.fit_generator(train_data, epochs=epochs, validation_data=val_data, verbose=1)

# Fine-tuning the model
if FINE_TUNE:
    for layer in base_model.layers[num_freeze_layers:]:
        layer.trainable = True
    print("=================================================================")
    print("Fine-tuning")
    print("=================================================================")
    model.compile(optimizer=RMSprop(lr=.001), loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    model.fit_generator(train_data, epochs=epochs, validation_data=val_data, verbose=1)

model.save('models/' + model_name + '.h5')
