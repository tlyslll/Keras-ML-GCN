from tensorflow.keras.layers import ConvLSTM2D,Conv2D,GlobalAvgPool2D,GlobalMaxPooling2D,ReLU,\
    LeakyReLU,Input,Lambda,Permute,Conv1D,GlobalMaxPooling1D,Reshape,GlobalAvgPool1D,BatchNormalization,\
    Layer,Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import Model
from graph import GraphConvolution
import tensorflow as tf


class Input_var(Layer):
    def __init__(self, input_shape):
        super(Input_var, self).__init__()

        w_init = tf.random_normal_initializer()

        self.w = tf.Variable(initial_value=w_init(shape=(input_shape[0], input_shape[1]),
                                                 dtype='float32'),
                            trainable=True)

        b_init = tf.zeros_initializer()

        self.b = tf.Variable(initial_value=b_init(shape=(1,),
                                                 dtype='float32'),
                            trainable=True)
    # 通过回调函数计算
    def call(self, inputs):
        return tf.multiply(inputs, self.w) + self.b # matmul 矩阵乘法

    def get_config(self):
        config={"input_shape":self.input_shape}
        base_config=super(Input_var, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def Conv_LeakyReLU(inputs,filters,kernel_size,strides=1):
    x=Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,padding="same")(inputs)
    x=Dropout(0.5)(x)
    x=BatchNormalization()(x)
    x=LeakyReLU(0.2)(x)
    return x

def GraphConv_LeakyReLU(inputs1,filters,support=1):
    x=GraphConvolution(units=filters,support=support)(inputs1)
    x=LeakyReLU(0.2)(x)
    return x

#ML_GCN
def LCAG_GAP(inputs_shape1, inputs_shape2,inputs_shape3,num_class):

    inputs1 = Input(inputs_shape1,name="image")
    inputs2 = Input(inputs_shape2,name="word_embding")
    inputs3=[Input(inputs_shape3,name="adj_martix")]
    inputs4=Reshape((7,7))(inputs3[0])

    x1 = Conv_LeakyReLU(inputs=inputs1, filters=128, kernel_size=3, strides=2)
    x1 = Conv_LeakyReLU(inputs=x1, filters=256, kernel_size=3)
    x1 = Conv_LeakyReLU(inputs=x1, filters=512, kernel_size=3, strides=2)
    x1 = Conv_LeakyReLU(inputs=x1, filters=1024, kernel_size=3)
    x1 = GlobalAvgPool2D()(x1)
    x2=GraphConv_LeakyReLU(inputs1=[inputs2[0]]+[inputs4[0]],filters=512)
    x2 = GraphConv_LeakyReLU(inputs1=[x2]+[inputs4], filters=1024)
    x2=Permute((2,1))(x2)
    x=Lambda(lambda x:tf.matmul(x[0],x[1]))([x1,x2])
    x=GlobalAvgPool1D()(x)
    x=Activation("sigmoid")(x)
    x = Model(inputs=[inputs1,inputs2]+inputs3, outputs=x)

    return x

#ML_GCN
def LCAG_GMP(inputs_shape1, inputs_shape2,inputs_shape3,num_class):

    inputs1 = Input(inputs_shape1,name="image")
    inputs2 = Input(inputs_shape2,name="word_embding")
    inputs3=[Input(inputs_shape3,name="adj_martix")]
    inputs4=Reshape((7,7))(inputs3[0])

    x1 = Conv_LeakyReLU(inputs=inputs1, filters=128, kernel_size=3, strides=2)
    x1 = Conv_LeakyReLU(inputs=x1, filters=256, kernel_size=3,strides=2)
    x1 = Conv_LeakyReLU(inputs=x1, filters=512, kernel_size=3, strides=2)
    x1 = Conv_LeakyReLU(inputs=x1, filters=1024, kernel_size=3,strides=2)
    x1 = GlobalMaxPooling2D()(x1)
    x2=Dropout(0.5)(inputs2[0])
    x2=GraphConv_LeakyReLU(inputs1=[x2]+[inputs4[0]],filters=512)
    x2=Dropout(0.5)(x2)
    x2 = GraphConv_LeakyReLU(inputs1=[x2]+[inputs4], filters=1024)

    x2=Permute((2,1))(x2)
    x=Lambda(lambda x:tf.matmul(x[0],x[1]))([x1,x2])

    x=GlobalMaxPooling1D()(x)
    x=Activation("sigmoid")(x)
    x = Model(inputs=[inputs1,inputs2]+inputs3, outputs=x)

    return x

#ML_GCN with word emdbing, input word vectors will follow the training

def LCAG_GMP_embding(inputs_shape1, inputs_shape2,inputs_shape3,num_class):

    inputs1 = Input(inputs_shape1,name="image")
    inputs2 = Input(inputs_shape2,name="word_embding")
    inputs3=[Input(inputs_shape3,name="adj_martix")]
    inputs4=Reshape((7,7))(inputs3[0])

    adj_martix=Input_var(input_shape=(7,7))(inputs4)

    x1 = Conv_LeakyReLU(inputs=inputs1, filters=512, kernel_size=3, strides=2)
    x1 = Conv_LeakyReLU(inputs=x1, filters=1024, kernel_size=3,strides=2)
    x1 = GlobalMaxPooling2D()(x1)

    x2=GraphConv_LeakyReLU(inputs1=[inputs2[0]]+[adj_martix[0]],filters=512)
    x2 = GraphConv_LeakyReLU(inputs1=[x2]+[adj_martix], filters=1024)

    x2=Permute((2,1))(x2)
    x=Lambda(lambda x:tf.matmul(x[0],x[1]))([x1,x2])
    x=GlobalMaxPooling1D()(x)
    x=Activation("sigmoid")(x)

    x = Model(inputs=[inputs1,inputs2]+inputs3, outputs=x)

    return x

