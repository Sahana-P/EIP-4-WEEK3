# EIP-4-WEEK3


1. Validation accracy for base model is 82.72   which is obtained at 49th   epoch

Epoch 00049: LearningRateScheduler setting learning rate to 0.00030652280529671407.
390/390 [==============================] - 22s 57ms/step - loss: 0.3536 - acc: 0.8756 - val_loss: 0.5173 - val_acc: 0.8272

2.weight_decay = 1e-4 ## https://appliedmachinelearning.blog/2018/03/24/achieving-90-accuracy-in-object-recognition-task-on-cifar-10-dataset-with-keras-convolutional-neural-networks/
model1 = Sequential()
model1.add(SeparableConv2D(filters= 32,kernel_size=(3,3),kernel_regularizer=regularizers.l2(weight_decay),depth_multiplier = 1,input_shape=(32,32,3),activation='relu'))  #30
model1.add(BatchNormalization())
model1.add(Dropout(0.2))
# 30*30*32
#receptive field =3
model1.add(SeparableConv2D(filters= 64,kernel_size=(3,3),kernel_regularizer=regularizers.l2(weight_decay),depth_multiplier = 1,activation='relu')) #28
model1.add(Dropout(0.2))
model1.add(BatchNormalization())
# 28*28*64
#receptive field =5
model1.add(SeparableConv2D(filters= 128,kernel_size=(3,3),kernel_regularizer=regularizers.l2(weight_decay),depth_multiplier = 1,activation='relu')) #26
model1.add(BatchNormalization())
model1.add(Dropout(0.2))
# 26*26*128
#receptive field =7
model1.add(MaxPooling2D(pool_size=(2, 2))) #13
# 13*13*32
#receptive field =8

model1.add(SeparableConv2D(filters= 128,kernel_size=(3,3),kernel_regularizer=regularizers.l2(weight_decay),depth_multiplier = 1,activation='relu')) #11
model1.add(BatchNormalization())
model1.add(Dropout(0.2))

# 11*11*128
#receptive field =12

model1.add(SeparableConv2D(filters= 256,kernel_size=(3,3),kernel_regularizer=regularizers.l2(weight_decay),depth_multiplier = 1,activation='relu')) #9
model1.add(BatchNormalization())
model1.add(Dropout(0.2))

# 9*9*256
#receptive field =16

model1.add(MaxPooling2D(pool_size=(2, 2))) 

# 4*4*256
#receptive field =18

model1.add(SeparableConv2D(filters= 64,kernel_size=(1,1),kernel_regularizer=regularizers.l2(weight_decay),depth_multiplier = 1,activation='relu')) 
model1.add(BatchNormalization())
model1.add(Dropout(0.2))

# 4*4*64
#receptive field =18

model1.add(SeparableConv2D(filters= 10,kernel_size=(1,1),kernel_regularizer=regularizers.l2(weight_decay),depth_multiplier = 1,activation='relu')) 
model1.add(BatchNormalization())
model1.add(Dropout(0.2))

# 4*4*10
#receptive field =18
model1.add(SeparableConv2D(filters= 10,kernel_size=(4,4),kernel_regularizer=regularizers.l2(weight_decay),depth_multiplier = 1,activation='relu')) 

# 1*1*10
#receptive field =30
model1.add(GlobalAveragePooling2D())
model1.add(Activation('softmax'))



3.Epoch 00001: LearningRateScheduler setting learning rate to 0.005.
  2/390 [..............................] - ETA: 25s - loss: 0.4758 - acc: 0.8281/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:11: UserWarning: The semantics of the Keras 2 argument `steps_per_epoch` is not the same as the Keras 1 argument `samples_per_epoch`. `steps_per_epoch` is the number of batches to draw from the generator at each epoch. Basically steps_per_epoch = samples_per_epoch/batch_size. Similarly `nb_val_samples`->`validation_steps` and `val_samples`->`steps` arguments have changed. Update your method calls accordingly.
  # This is added back by InteractiveShellApp.init_path()
/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:11: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<keras_pre..., validation_data=(array([[[..., callbacks=[<keras.ca..., steps_per_epoch=390, epochs=50)`
  # This is added back by InteractiveShellApp.init_path()
390/390 [==============================] - 23s 58ms/step - loss: 0.5613 - acc: 0.8044 - val_loss: 0.7720 - val_acc: 0.7484
Epoch 2/50

Epoch 00002: LearningRateScheduler setting learning rate to 0.0037907505686125857.
390/390 [==============================] - 23s 58ms/step - loss: 0.5324 - acc: 0.8132 - val_loss: 0.6453 - val_acc: 0.7789
Epoch 3/50

Epoch 00003: LearningRateScheduler setting learning rate to 0.003052503052503053.
390/390 [==============================] - 22s 58ms/step - loss: 0.5085 - acc: 0.8220 - val_loss: 0.5937 - val_acc: 0.8023
Epoch 4/50

Epoch 00004: LearningRateScheduler setting learning rate to 0.0025549310168625446.
390/390 [==============================] - 23s 58ms/step - loss: 0.4894 - acc: 0.8280 - val_loss: 0.8031 - val_acc: 0.7355
Epoch 5/50

Epoch 00005: LearningRateScheduler setting learning rate to 0.0021968365553602814.
390/390 [==============================] - 22s 58ms/step - loss: 0.4716 - acc: 0.8359 - val_loss: 0.6602 - val_acc: 0.7724
Epoch 6/50

Epoch 00006: LearningRateScheduler setting learning rate to 0.001926782273603083.
390/390 [==============================] - 23s 58ms/step - loss: 0.4628 - acc: 0.8375 - val_loss: 0.6427 - val_acc: 0.7830
Epoch 7/50

Epoch 00007: LearningRateScheduler setting learning rate to 0.0017158544955387784.
390/390 [==============================] - 23s 58ms/step - loss: 0.4540 - acc: 0.8415 - val_loss: 0.6119 - val_acc: 0.7909
Epoch 8/50

Epoch 00008: LearningRateScheduler setting learning rate to 0.0015465511908444168.
390/390 [==============================] - 22s 58ms/step - loss: 0.4442 - acc: 0.8448 - val_loss: 0.5818 - val_acc: 0.8064
Epoch 9/50

Epoch 00009: LearningRateScheduler setting learning rate to 0.0014076576576576576.
390/390 [==============================] - 23s 58ms/step - loss: 0.4357 - acc: 0.8477 - val_loss: 0.5507 - val_acc: 0.8145
Epoch 10/50

Epoch 00010: LearningRateScheduler setting learning rate to 0.0012916559028674762.
390/390 [==============================] - 23s 58ms/step - loss: 0.4292 - acc: 0.8512 - val_loss: 0.5459 - val_acc: 0.8146
Epoch 11/50

Epoch 00011: LearningRateScheduler setting learning rate to 0.0011933174224343678.
390/390 [==============================] - 22s 58ms/step - loss: 0.4191 - acc: 0.8531 - val_loss: 0.5533 - val_acc: 0.8124
Epoch 12/50

Epoch 00012: LearningRateScheduler setting learning rate to 0.0011088933244621866.
390/390 [==============================] - 23s 58ms/step - loss: 0.4179 - acc: 0.8535 - val_loss: 0.5448 - val_acc: 0.8168
Epoch 13/50

Epoch 00013: LearningRateScheduler setting learning rate to 0.001035625517812759.
390/390 [==============================] - 22s 58ms/step - loss: 0.4108 - acc: 0.8559 - val_loss: 0.5518 - val_acc: 0.8161
Epoch 14/50

Epoch 00014: LearningRateScheduler setting learning rate to 0.0009714396735962696.
390/390 [==============================] - 23s 58ms/step - loss: 0.4107 - acc: 0.8565 - val_loss: 0.5443 - val_acc: 0.8172
Epoch 15/50

Epoch 00015: LearningRateScheduler setting learning rate to 0.0009147457006952067.
390/390 [==============================] - 23s 58ms/step - loss: 0.4035 - acc: 0.8591 - val_loss: 0.5375 - val_acc: 0.8191
Epoch 16/50

Epoch 00016: LearningRateScheduler setting learning rate to 0.000864304235090752.
390/390 [==============================] - 23s 58ms/step - loss: 0.3988 - acc: 0.8595 - val_loss: 0.5589 - val_acc: 0.8092
Epoch 17/50

Epoch 00017: LearningRateScheduler setting learning rate to 0.00081913499344692.
390/390 [==============================] - 23s 58ms/step - loss: 0.4007 - acc: 0.8595 - val_loss: 0.5637 - val_acc: 0.8078
Epoch 18/50

Epoch 00018: LearningRateScheduler setting learning rate to 0.0007784524365561264.
390/390 [==============================] - 22s 58ms/step - loss: 0.3962 - acc: 0.8597 - val_loss: 0.5520 - val_acc: 0.8156
Epoch 19/50

Epoch 00019: LearningRateScheduler setting learning rate to 0.0007416196974191634.
390/390 [==============================] - 23s 58ms/step - loss: 0.3868 - acc: 0.8645 - val_loss: 0.5759 - val_acc: 0.8071
Epoch 20/50

Epoch 00020: LearningRateScheduler setting learning rate to 0.0007081149978756551.
390/390 [==============================] - 22s 58ms/step - loss: 0.3949 - acc: 0.8608 - val_loss: 0.5185 - val_acc: 0.8240
Epoch 21/50

Epoch 00021: LearningRateScheduler setting learning rate to 0.0006775067750677507.
390/390 [==============================] - 23s 58ms/step - loss: 0.3910 - acc: 0.8634 - val_loss: 0.5408 - val_acc: 0.8143
Epoch 22/50

Epoch 00022: LearningRateScheduler setting learning rate to 0.0006494349915573452.
390/390 [==============================] - 23s 58ms/step - loss: 0.3867 - acc: 0.8643 - val_loss: 0.5470 - val_acc: 0.8149
Epoch 23/50

Epoch 00023: LearningRateScheduler setting learning rate to 0.0006235969069593414.
390/390 [==============================] - 23s 58ms/step - loss: 0.3754 - acc: 0.8690 - val_loss: 0.5349 - val_acc: 0.8200
Epoch 24/50

Epoch 00024: LearningRateScheduler setting learning rate to 0.0005997361161089121.
390/390 [==============================] - 23s 58ms/step - loss: 0.3835 - acc: 0.8656 - val_loss: 0.5398 - val_acc: 0.8185
Epoch 25/50

Epoch 00025: LearningRateScheduler setting learning rate to 0.000577634011090573.
390/390 [==============================] - 22s 57ms/step - loss: 0.3762 - acc: 0.8673 - val_loss: 0.5344 - val_acc: 0.8204
Epoch 26/50

Epoch 00026: LearningRateScheduler setting learning rate to 0.0005571030640668523.
390/390 [==============================] - 22s 58ms/step - loss: 0.3746 - acc: 0.8683 - val_loss: 0.5231 - val_acc: 0.8250
Epoch 27/50

Epoch 00027: LearningRateScheduler setting learning rate to 0.0005379814934366257.
390/390 [==============================] - 22s 58ms/step - loss: 0.3809 - acc: 0.8665 - val_loss: 0.5461 - val_acc: 0.8164
Epoch 28/50

Epoch 00028: LearningRateScheduler setting learning rate to 0.0005201289919900136.
390/390 [==============================] - 23s 58ms/step - loss: 0.3764 - acc: 0.8698 - val_loss: 0.5465 - val_acc: 0.8159
Epoch 29/50

Epoch 00029: LearningRateScheduler setting learning rate to 0.0005034232782923882.
390/390 [==============================] - 23s 58ms/step - loss: 0.3749 - acc: 0.8677 - val_loss: 0.5253 - val_acc: 0.8232
Epoch 30/50

Epoch 00030: LearningRateScheduler setting learning rate to 0.000487757291971515.
390/390 [==============================] - 23s 58ms/step - loss: 0.3708 - acc: 0.8700 - val_loss: 0.5381 - val_acc: 0.8198
Epoch 31/50

Epoch 00031: LearningRateScheduler setting learning rate to 0.0004730368968779565.
390/390 [==============================] - 23s 58ms/step - loss: 0.3731 - acc: 0.8697 - val_loss: 0.5279 - val_acc: 0.8256
Epoch 32/50

Epoch 00032: LearningRateScheduler setting learning rate to 0.00045917898796951054.
390/390 [==============================] - 22s 57ms/step - loss: 0.3676 - acc: 0.8700 - val_loss: 0.5364 - val_acc: 0.8206
Epoch 33/50

Epoch 00033: LearningRateScheduler setting learning rate to 0.0004461099214846538.
390/390 [==============================] - 22s 58ms/step - loss: 0.3676 - acc: 0.8712 - val_loss: 0.5453 - val_acc: 0.8169
Epoch 34/50

Epoch 00034: LearningRateScheduler setting learning rate to 0.0004337642057777392.
390/390 [==============================] - 22s 57ms/step - loss: 0.3666 - acc: 0.8710 - val_loss: 0.5288 - val_acc: 0.8221
Epoch 35/50

Epoch 00035: LearningRateScheduler setting learning rate to 0.0004220834036805673.
390/390 [==============================] - 23s 58ms/step - loss: 0.3722 - acc: 0.8689 - val_loss: 0.5266 - val_acc: 0.8232
Epoch 36/50

Epoch 00036: LearningRateScheduler setting learning rate to 0.0004110152075626798.
390/390 [==============================] - 22s 58ms/step - loss: 0.3684 - acc: 0.8698 - val_loss: 0.5223 - val_acc: 0.8232
Epoch 37/50

Epoch 00037: LearningRateScheduler setting learning rate to 0.0004005126561999359.
390/390 [==============================] - 22s 58ms/step - loss: 0.3644 - acc: 0.8719 - val_loss: 0.5256 - val_acc: 0.8230
Epoch 38/50

Epoch 00038: LearningRateScheduler setting learning rate to 0.00039053346871826915.
390/390 [==============================] - 22s 58ms/step - loss: 0.3653 - acc: 0.8715 - val_loss: 0.5243 - val_acc: 0.8249
Epoch 39/50

Epoch 00039: LearningRateScheduler setting learning rate to 0.00038103947568968145.
390/390 [==============================] - 23s 58ms/step - loss: 0.3604 - acc: 0.8727 - val_loss: 0.5262 - val_acc: 0.8222
Epoch 40/50

Epoch 00040: LearningRateScheduler setting learning rate to 0.0003719961312402351.
390/390 [==============================] - 23s 58ms/step - loss: 0.3643 - acc: 0.8719 - val_loss: 0.5279 - val_acc: 0.8234
Epoch 41/50

Epoch 00041: LearningRateScheduler setting learning rate to 0.0003633720930232558.
390/390 [==============================] - 23s 58ms/step - loss: 0.3573 - acc: 0.8738 - val_loss: 0.5307 - val_acc: 0.8257
Epoch 42/50

Epoch 00042: LearningRateScheduler setting learning rate to 0.00035513885929398396.
390/390 [==============================] - 23s 58ms/step - loss: 0.3590 - acc: 0.8725 - val_loss: 0.5138 - val_acc: 0.8251
Epoch 43/50

Epoch 00043: LearningRateScheduler setting learning rate to 0.0003472704542297541.
390/390 [==============================] - 23s 58ms/step - loss: 0.3616 - acc: 0.8733 - val_loss: 0.5241 - val_acc: 0.8260
Epoch 44/50

Epoch 00044: LearningRateScheduler setting learning rate to 0.00033974315417544334.
390/390 [==============================] - 23s 58ms/step - loss: 0.3582 - acc: 0.8731 - val_loss: 0.5331 - val_acc: 0.8193
Epoch 45/50

Epoch 00045: LearningRateScheduler setting learning rate to 0.00033253524873636604.
390/390 [==============================] - 22s 58ms/step - loss: 0.3564 - acc: 0.8749 - val_loss: 0.5249 - val_acc: 0.8222
Epoch 46/50

Epoch 00046: LearningRateScheduler setting learning rate to 0.00032562683165092806.
390/390 [==============================] - 23s 58ms/step - loss: 0.3578 - acc: 0.8738 - val_loss: 0.5255 - val_acc: 0.8244
Epoch 47/50

Epoch 00047: LearningRateScheduler setting learning rate to 0.0003189996172004594.
390/390 [==============================] - 22s 58ms/step - loss: 0.3573 - acc: 0.8748 - val_loss: 0.5270 - val_acc: 0.8225
Epoch 48/50

Epoch 00048: LearningRateScheduler setting learning rate to 0.0003126367785906334.
390/390 [==============================] - 23s 58ms/step - loss: 0.3579 - acc: 0.8730 - val_loss: 0.5199 - val_acc: 0.8255
Epoch 49/50

Epoch 00049: LearningRateScheduler setting learning rate to 0.00030652280529671407.
390/390 [==============================] - 22s 57ms/step - loss: 0.3536 - acc: 0.8756 - val_loss: 0.5173 - val_acc: 0.8272
Epoch 50/50

Epoch 00050: LearningRateScheduler setting learning rate to 0.0003006433768264085.
390/390 [==============================] - 23s 58ms/step - loss: 0.3539 - acc: 0.8761 - val_loss: 0.5218 - val_acc: 0.8249
