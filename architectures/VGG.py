from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


class VGGTypeA:

    @staticmethod
    def build():
        model = Sequential()
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', input_shape=(224, 224, 3),
                         activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same',
                         activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same',
                         activation='relu'))
        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same',
                         activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same',
                         activation='relu'))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same',
                         activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same',
                         activation='relu'))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same',
                         activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        return model

m = VGGTypeA.build()
m.summary()