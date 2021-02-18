from tensorflow.keras.activations import softmax
from tensorflow.keras.layers import Add, Concatenate, Conv2D, Input, MaxPooling2D, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

def fire_module(s1, e1, e3):
    """s1 - number of 1x1 filters in squeeze layer\n
       e1 - number of 1x1 filters in expand layer\n
       e3 - number of 3x3 filters in expand layer\n
       advise: s1 < e1 + e3"""
    def fire_layer(output):
        squeeze_layer = Conv2D(filters=s1, kernel_size=1, activation='relu')(output)
        expand_1 = Conv2D(filters=e1, kernel_size=1, activation='relu')(squeeze_layer)
        expand_3 = Conv2D(filters=e3, kernel_size=3, activation='relu', padding='same')(squeeze_layer)
        final = Concatenate()([expand_1, expand_3])
        return final
    return fire_layer


def squeeze_net(shape):
    inp = Input(shape=shape)
    conv1 = Conv2D(filters=96, kernel_size=(7, 7), strides=(2, 2), activation='relu',
                   padding='same')(inp)
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv1)
    fire1 = fire_module(16, 64, 64)(pool1)
    fire2 = fire_module(16, 64, 64)(fire1)
    fire3 = fire_module(32, 128, 128)(fire2)
    pool2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(fire3)
    fire4 = fire_module(32, 128, 128)(pool2)
    fire5 = fire_module(48, 192, 192)(fire4)
    fire6 = fire_module(48, 192, 192)(fire5)
    fire7 = fire_module(64, 256, 256)(fire6)
    pool3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(fire7)
    fire8 = fire_module(64, 256, 256)(pool3)
    drop1 = Dropout(.5)(fire8)

    # conv2 should have number of filters same as desired number of outputs
    conv2 = Conv2D(filters=1000, kernel_size=(1, 1), strides=(1, 1), activation='relu')(drop1)
    pool4 = GlobalAveragePooling2D()(conv2)

    # activation functions should be changed to switch to regression
    out = softmax(pool4)

    model = Model(inputs=inp, outputs=[out])
    return model


def squeeze_net_simple_skip(shape):
    inp = Input(shape=shape)
    conv1 = Conv2D(filters=96, kernel_size=(7, 7), strides=(2, 2), activation='relu',
                   padding='same')(inp)
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv1)
    fire1 = fire_module(16, 64, 64)(pool1)
    fire2 = fire_module(16, 64, 64)(fire1)
    add1 = Add()([fire1, fire2])
    fire3 = fire_module(32, 128, 128)(add1)
    pool2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(fire3)
    fire4 = fire_module(32, 128, 128)(pool2)
    add2 = Add()([pool2, fire4])
    fire5 = fire_module(48, 192, 192)(add2)
    fire6 = fire_module(48, 192, 192)(fire5)
    add3 = Add()([fire5, fire6])
    fire7 = fire_module(64, 256, 256)(add3)
    pool3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(fire7)
    fire8 = fire_module(64, 256, 256)(pool3)
    add4 = Add()([pool3, fire8])
    drop1 = Dropout(.5)(add4)

    # conv2 should have number of filters same as desired number of outputs
    conv2 = Conv2D(filters=1000, kernel_size=(1, 1), strides=(1, 1), activation='relu')(drop1)
    pool4 = GlobalAveragePooling2D()(conv2)

    # activation functions should be changed to switch to regression
    out = softmax(pool4)

    model = Model(inputs=inp, outputs=[out])
    return model


def squeeze_net_complex_skip(shape):
    inp = Input(shape=shape)
    conv1 = Conv2D(filters=96, kernel_size=(7, 7), strides=(2, 2), activation='relu',
                   padding='same')(inp)
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv1)
    skip_conv1 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), activation='relu')(pool1)
    fire1 = fire_module(16, 64, 64)(pool1)
    complex_add1 = Add()([fire1, skip_conv1])
    fire2 = fire_module(16, 64, 64)(complex_add1)
    add1 = Add()([complex_add1, fire2])
    fire3 = fire_module(32, 128, 128)(add1)
    skip_conv2 = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), activation='relu')(add1)
    complex_add2 = Add()([skip_conv2, fire3])
    pool2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(complex_add2)
    fire4 = fire_module(32, 128, 128)(pool2)
    add2 = Add()([pool2, fire4])
    skip_conv3 = Conv2D(filters=384, kernel_size=(1, 1), strides=(1, 1), activation='relu')(add2)
    fire5 = fire_module(48, 192, 192)(add2)
    complex_add3 = Add()([skip_conv3, fire5])
    fire6 = fire_module(48, 192, 192)(complex_add3)
    add3 = Add()([complex_add3, fire6])
    fire7 = fire_module(64, 256, 256)(add3)
    skip_conv4 = Conv2D(filters=512, kernel_size=(1, 1), strides=(1, 1), activation='relu')(add3)
    complex_add4 = Add()([skip_conv4, fire7])
    pool3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(complex_add4)
    fire8 = fire_module(64, 256, 256)(pool3)
    add4 = Add()([pool3, fire8])
    drop1 = Dropout(.5)(add4)

    # conv2 should have number of filters same as desired number of outputs
    conv2 = Conv2D(filters=1000, kernel_size=(1, 1), strides=(1, 1), activation='relu')(drop1)
    pool4 = GlobalAveragePooling2D()(conv2)

    # activation functions should be changed to switch to regression
    out = softmax(pool4)

    model = Model(inputs=inp, outputs=[out])
    return model


m = squeeze_net_complex_skip((224, 224, 3))
m.summary()

plot_model(m)








