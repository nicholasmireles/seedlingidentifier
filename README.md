# Keras TensorFlow Xception Seedling Identifier
##### v1.0
###### Created by Nicholas Mireles 6/14/2018
This is my attempt at creating an Xception-based model for the recognition of seedling images.

### Current attempt
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
xception (Model)             (None, None, None, 2048)  20861480  
_________________________________________________________________
global_average_pooling2d_5 ( (None, 2048)              0         
_________________________________________________________________
dense_5 (Dense)              (None, 12)                24588     
_________________________________________________________________
dropout_4 (Dropout)          (None, 12)                0         
=================================================================
Total params: 20,886,068
Trainable params: 24,588
Non-trainable params: 20,861,480
_________________________________________________________________
None
Epoch 1/12
30/30 [==============================] - 36s 1s/step - loss: 4.2969 - acc: 0.3378 - val_loss: 2.2931 - val_acc: 0.2862
Epoch 2/12
30/30 [==============================] - 27s 912ms/step - loss: 3.8237 - acc: 0.5174 - val_loss: 2.2259 - val_acc: 0.2914
Epoch 3/12
30/30 [==============================] - 29s 951ms/step - loss: 3.8594 - acc: 0.5830 - val_loss: 2.0733 - val_acc: 0.3453
Epoch 4/12
30/30 [==============================] - 29s 951ms/step - loss: 3.6191 - acc: 0.6236 - val_loss: 2.1851 - val_acc: 0.2988
Epoch 5/12
30/30 [==============================] - 28s 932ms/step - loss: 3.5435 - acc: 0.6474 - val_loss: 2.1222 - val_acc: 0.3559
Epoch 6/12
30/30 [==============================] - 28s 933ms/step - loss: 3.4465 - acc: 0.6552 - val_loss: 2.1730 - val_acc: 0.3369
Epoch 7/12
30/30 [==============================] - 27s 910ms/step - loss: 3.5021 - acc: 0.6646 - val_loss: 2.1678 - val_acc: 0.3379
Epoch 8/12
30/30 [==============================] - 28s 935ms/step - loss: 3.4276 - acc: 0.6700 - val_loss: 2.1683 - val_acc: 0.3369
Epoch 9/12
30/30 [==============================] - 28s 930ms/step - loss: 3.4097 - acc: 0.6837 - val_loss: 2.1963 - val_acc: 0.3284
Epoch 10/12
30/30 [==============================] - 28s 938ms/step - loss: 3.3898 - acc: 0.6878 - val_loss: 2.2527 - val_acc: 0.3347
Epoch 11/12
30/30 [==============================] - 28s 921ms/step - loss: 3.4416 - acc: 0.6816 - val_loss: 2.2296 - val_acc: 0.3305
Epoch 12/12
30/30 [==============================] - 28s 922ms/step - loss: 3.4668 - acc: 0.6870 - val_loss: 2.2174 - val_acc: 0.3495
=================================================================
Fine-tuning
=================================================================
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
xception (Model)             (None, None, None, 2048)  20861480  
_________________________________________________________________
global_average_pooling2d_5 ( (None, 2048)              0         
_________________________________________________________________
dense_5 (Dense)              (None, 12)                24588     
_________________________________________________________________
dropout_4 (Dropout)          (None, 12)                0         
=================================================================
Total params: 20,886,068
Trainable params: 4,773,388
Non-trainable params: 16,112,680
_________________________________________________________________
None
Epoch 1/12
30/30 [==============================] - 36s 1s/step - loss: 3.1451 - acc: 0.7288 - val_loss: 2.7776 - val_acc: 0.3390
Epoch 2/12
30/30 [==============================] - 29s 958ms/step - loss: 2.9197 - acc: 0.7847 - val_loss: 2.9034 - val_acc: 0.3696
Epoch 3/12
30/30 [==============================] - 29s 958ms/step - loss: 2.8820 - acc: 0.7991 - val_loss: 3.5511 - val_acc: 0.3178
Epoch 4/12
30/30 [==============================] - 29s 962ms/step - loss: 2.5921 - acc: 0.8250 - val_loss: 2.8373 - val_acc: 0.3749
Epoch 5/12
30/30 [==============================] - 29s 962ms/step - loss: 2.5327 - acc: 0.8344 - val_loss: 2.5394 - val_acc: 0.3970
Epoch 6/12
30/30 [==============================] - 29s 961ms/step - loss: 2.7191 - acc: 0.8301 - val_loss: 2.8713 - val_acc: 0.3633
Epoch 7/12
30/30 [==============================] - 29s 959ms/step - loss: 2.4660 - acc: 0.8458 - val_loss: 2.8329 - val_acc: 0.3643
Epoch 8/12
30/30 [==============================] - 29s 958ms/step - loss: 2.6583 - acc: 0.8347 - val_loss: 2.8277 - val_acc: 0.4034
Epoch 9/12
30/30 [==============================] - 29s 965ms/step - loss: 2.7524 - acc: 0.8337 - val_loss: 2.9992 - val_acc: 0.3453
Epoch 10/12
30/30 [==============================] - 29s 958ms/step - loss: 2.5675 - acc: 0.8436 - val_loss: 2.8985 - val_acc: 0.3770
Epoch 11/12
30/30 [==============================] - 29s 962ms/step - loss: 2.5384 - acc: 0.8430 - val_loss: 2.4411 - val_acc: 0.4456
Epoch 12/12
30/30 [==============================] - 29s 965ms/step - loss: 2.8099 - acc: 0.8313 - val_loss: 2.3620 - val_acc: 0.4287
```
### Current Accuracy
Validation:

Test:

### Changelog
###### 6/14/18
- Repo created!
