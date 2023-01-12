import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import SGD,Adam
from models import LCAG_GMP,LCAG_GMP_embding
import time
import scipy.sparse as sp
import os
from data_generator import DataGenerator
import matplotlib.pyplot as plt


os.environ["CUDA_VISIBLE_DEVICES"]="0"

if __name__ == "__main__":

    batch_size=2
    # 模型保存的位置
    log_dir = "./logs/"
    # 打开数据集的txt
    with open(r"train\train4.txt","r") as f:
        lines = f.readlines()
    data_list=[]
    for i in range(len(lines)):
        data_list.append(lines[i].strip("\n").split(" "))
    data_list=np.array(data_list)
    data_list=data_list.reshape((len(lines),5))

    # 90%用于训练，10%用于估计
    num_val = int(len(lines)*0.1)
    num_train = len(lines) - num_val
    #print(num_train)

    model = LCAG_GMP(inputs_shape1=(224,224,3),inputs_shape2=(7,300),inputs_shape3=(49,),num_class=7)
    #model=LCAG_GMP_embding(inputs_shape1=(224,224,3),inputs_shape2=(7,300),inputs_shape3=(49,),num_class=7)
    print(model.summary())

    # 保存的方式，3代保存一次
    checkpoint_period = ModelCheckpoint(
        log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='acc',
        save_weights_only=False,
        save_best_only=True,
        period=3
    )
    # 学习率下降的方式，acc三次不下降就下降学习率继续训练
    reduce_lr = ReduceLROnPlateau(
        monitor='acc',
        factor=0.5,
        patience=3,
        verbose=1
    )
    # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=10,
        verbose=1
    )

    model.compile(loss = 'categorical_crossentropy',
            optimizer = SGD(lr=1e-3,momentum=0.9),
            metrics = ['accuracy'])

    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    train_generator = DataGenerator(data_list[:num_train], batch_size)
    varl_generator = DataGenerator(data_list[num_train:], batch_size)
    history=model.fit(train_generator,
              steps_per_epoch=max(1, num_train // batch_size),
              validation_data=varl_generator,
              validation_steps=max(1, num_val // batch_size),
              epochs=100,
              initial_epoch=0,
              callbacks=[checkpoint_period, reduce_lr,early_stopping],
              verbose=1)
    model.save_weights(log_dir + 'last1.h5')

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, ls='-', color='purple', lw=1, label='Training accuracy')
    plt.plot(epochs, val_acc, ls='-', color='red',lw=1, label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, ls='-', color='purple', lw=1,label='Training loss')
    plt.plot(epochs, val_loss, ls='-', color='red', lw=1,label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()