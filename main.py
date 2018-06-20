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
CALCULATE_BN = False

# File related variables
train_dir = 'data/train/'
test_dir = 'data/test/'
bot_dir = 'data/bottleneck/'
model_name = 'Inception'

# Defining some parameters for the model
batch_size = 128
num_epochs = 30
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
train_data = train_gen.flow_from_directory(train_dir, subset="training", batch_size=batch_size, shuffle=False,target_size=(299,299))
val_data = train_gen.flow_from_directory(train_dir, subset="validation", batch_size=batch_size, shuffle=False,target_size=(299,299))
test_data = test_gen.flow_from_directory(test_dir, batch_size=batch_size, class_mode=None)


if CALCULATE_BN:
    # Loading the Xception base model
    print('Loading base model...')
    # base_model = Xception(weights='imagenet',include_top=False,pooling=pooling)
    base_model = InceptionV3(weights='imagenet', include_top=False, pooling=pooling)

    print('Calculating bottleneck features for dataset.')

    X_train = np.zeros((0, 2048))
    y_train = np.zeros((0, 12))
    X_val = np.zeros((0, 2048))
    y_val = np.zeros((0, 12))

    num_batches = len(train_data)
    i = 1
    while i <= num_batches:
        print('Working on training batch: {}/{}'.format(i, num_batches))
        batch_x, batch_y = train_data.next()
        train_predictions = base_model.predict(batch_x)
        X_train = np.concatenate((X_train, train_predictions), axis=0)
        y_train = np.concatenate((y_train, batch_y), axis=0)
        i += 1

    print('Saving training features to disk.')
    np.save(bot_dir + model_name + '_trainscores.npy', X_train)
    np.save(bot_dir + model_name + '_trainlabels.npy', y_train)

    num_batches = len(val_data)
    i = 1
    while i <= num_batches:
        print('Working on validation batch: {}/{}'.format(i, num_batches))
        batch_x, batch_y = val_data.next()
        val_predictions = base_model.predict(batch_x)
        X_val = np.concatenate((X_val, val_predictions), axis=0)
        y_val = np.concatenate((y_val, batch_y), axis=0)
        i += 1

    print('Saving validation features to disk.')
    np.save(bot_dir + model_name + '_valscores.npy', X_val)
    np.save(bot_dir + model_name + '_vallabels.npy', y_val)

else:
    print('Loading bottleneck scores from disk.')
    X_train = np.load(bot_dir + model_name + '_trainscores.npy')
    y_train = np.load(bot_dir + model_name + '_trainlabels.npy')
    X_val = np.load(bot_dir + model_name + '_valscores.npy')
    y_val = np.load(bot_dir + model_name + '_vallabels.npy')


# Adding in the new layers
model = Sequential()
# model.add(Dense(dense1,activation="relu"))
# model.add(Dropout(.5))
model.add(Dense(dense2,input_shape=(2048,), activation="softmax", kernel_regularizer=l2(0.01)))

# Compiling/fitting the model for transfer-learning
model.compile(optimizer="Adadelta", loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())
model.fit(x=X_train,y=y_train,batch_size=batch_size,epochs=num_epochs,verbose=1,validation_data=(X_val,y_val))

# Fine-tuning the model
if FINE_TUNE:
    # Loading the Xception base model
    print('Loading base model...')
    # base_model = Xception(weights='imagenet',include_top=False,pooling=pooling)
    base_model = InceptionV3(weights='imagenet', include_top=False, pooling=pooling)
    print(base_model.input)
    for layer in base_model.layers[num_freeze_layers:]:
        layer.trainable = True

    new_model = Sequential()
    new_model.add(base_model)
    new_model.add(model)

    print("=================================================================")
    print("Fine-tuning")
    print("=================================================================")
    new_model.compile(optimizer=RMSprop(lr=.001), loss='categorical_crossentropy', metrics=['accuracy'])
    print(new_model.summary())
    new_model.fit_generator(train_data, epochs=num_epochs, validation_data=val_data, verbose=1)
    new_model.save('models/' + model_name + '.h5')
else:
    model.save('models/' + model_name + '.h5')
