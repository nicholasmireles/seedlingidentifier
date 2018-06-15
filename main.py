import tensorflow
import sys
from keras.applications.xception import Xception,preprocess_input
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from keras.regularizers import l2
# File related variables
train_dir = 'data/train/'
test_dir = 'data/test/'
model_name = 'Xception1.h5'

# Defining some parameters for the model
batch_size= 128
epochs = 12
dense1 = 32
dense2 = 12
pooling = None
fine_tune = True
num_freeze_layers = 126

# Creating the generators for the images
train_gen = image.ImageDataGenerator(validation_split=.2,preprocessing_function=preprocess_input)
test_gen = image.ImageDataGenerator(preprocessing_function=preprocess_input)

# Creating the actual data generator objects for the different sets
train_data = train_gen.flow_from_directory(train_dir,subset="training",batch_size=batch_size)
val_data = train_gen.flow_from_directory(train_dir,subset="validation",batch_size=batch_size)
test_data = test_gen.flow_from_directory(test_dir,batch_size=batch_size,class_mode=None)

# Loading the Xception base model
print('Loading base model...')
base_model = Xception(weights='imagenet',include_top=False,pooling=pooling)

# Freezing weights for transfer-learning:
for layer in base_model.layers:
    layer.trainable = False

# Adding in the new layers
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
#model.add(Dense(dense1,activation="relu"))
#model.add(Dropout(.5))
model.add(Dense(dense2,activation="softmax",kernel_regularizer=l2(0.01)))
model.add(Dropout(.15))

#Compiling/fitting the model for transfer-learning
model.compile(optimizer="Adadelta",loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())
model.fit_generator(train_data,epochs=epochs,validation_data=val_data,verbose=1)

# Fine-tuning the model
if fine_tune:
    for layer in base_model.layers[num_freeze_layers:]:
        layer.trainable = True
    print("=================================================================")
    print("Fine-tuning")
    print("=================================================================")
    model.compile(optimizer="Adadelta",loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    model.fit_generator(train_data,epochs=epochs,validation_data=val_data,verbose=1)

model.save('models/'+model_name)
