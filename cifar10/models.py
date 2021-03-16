import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Add, Conv2D, Dropout, Flatten, Dense, MaxPooling2D, GlobalAveragePooling2D, Lambda, BatchNormalization, concatenate, DepthwiseConv2D, ReLU, Activation, Multiply
from tensorflow.keras import regularizers
from tensorflow.keras.activations import softmax
from tensorflow import keras


class VGG(object):
    def __init__(self, num_classes, blocks=[(3, 32)]):
        self._num_classes = num_classes
        self._blocks = blocks

    @staticmethod
    def vgg_block(x, num_convs, num_filters):
        out = x
        for _ in range(num_convs):
            out = Conv2D(num_filters, kernel_size=(3, 3), activation="relu", padding="SAME")(out)
        out = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(out)
        return out

    def _build_graph(self):
        inputs = Input((32, 32, 3))
        outputs = inputs

        for num_convs, num_filters in self._blocks:
            outputs = self.vgg_block(outputs, num_convs, num_filters)

        outputs = Flatten()(outputs)
        outputs = Dense(128, activation="relu")(outputs)
        outputs = Dense(self._num_classes, activation="softmax")(outputs)
        return inputs, outputs

    def get(self):
        inputs, outputs = self._build_graph()
        self._model = keras.Model(inputs=inputs, outputs=outputs)
        return self._model


class SqueezeNet(object):
    def __init__(self, num_classes, groups=[]):
        self._num_classes = num_classes
        self._groups = groups

    @staticmethod
    def fire_module(x, num_squeeze_filters, num_expand_filters):
        out = x
        squeezed = Conv2D(num_squeeze_filters, kernel_size=(1, 1), activation="relu", padding="valid")(x)
        expand_1x1 = Conv2D(num_expand_filters, kernel_size=(1, 1), activation="relu", padding="valid")(squeezed)
        expand_3x3 = Conv2D(num_expand_filters, kernel_size=(3, 3), activation="relu", padding="SAME")(squeezed)
        return concatenate((expand_1x1, expand_3x3), axis=3)

    def _build_graph(self):
        inputs = Input((32, 32, 3))
        outputs = Conv2D(32, kernel_size=(3, 3),strides=(2, 2), activation="relu", padding="valid")(inputs)
        outputs = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(outputs)
        for group in self._groups:
            for num_fires, num_squeeze_filters, num_expand_filters in group:
                for _ in range(num_fires):
                    outputs = self.fire_module(outputs, num_squeeze_filters, num_expand_filters)

            outputs = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(outputs)

        outputs = Conv2D(self._num_classes, kernel_size=(1, 1), activation="relu", padding="valid")(outputs)
        outputs = GlobalAveragePooling2D()(outputs)
        outputs = softmax(outputs)
        return inputs, outputs

    def get(self):
        inputs, outputs = self._build_graph()
        self._model = keras.Model(inputs=inputs, outputs=outputs)
        return self._model


class SqueezeNext(object):
    def __init__(self, num_classes, groups=[]):
        self._num_classes = num_classes
        self._groups = groups

    @staticmethod
    def bottleneck_module(inputs, num_output_filters):
        input_channels = x.get_shape()[-1];
        outputs = x
        step1 = Conv2D(input_channels / 2, kernel_size=(1, 1), activation="relu", padding="same")(x)
        step2 = Conv2D(input_channels / 4, kernel_size=(1, 1), activation="relu", padding="same")(step1)

    def _build_graph(self):
        inputs = Input((32, 32, 3))
        outputs = Conv2D(32, kernel_size=(3, 3),strides=(2, 2), activation="relu", padding="valid")(inputs)
        outputs = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(outputs)
        for group in self._groups:
            for num_fires, num_squeeze_filters, num_expand_filters in group:
                for _ in range(num_fires):
                    outputs = self.fire_module(outputs, num_squeeze_filters, num_expand_filters)

            outputs = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(outputs)

        outputs = Conv2D(self._num_classes, kernel_size=(1, 1), activation="relu", padding="valid")(outputs)
        outputs = GlobalAveragePooling2D()(outputs)
        outputs = softmax(outputs)
        return inputs, outputs

    def get(self):
        inputs, outputs = self._build_graph()
        self._model = keras.Model(inputs=inputs, outputs=outputs)
        return self._model

class MobileNetV1(object):
    def __init__(self, num_classes, blocks=[]):
        self._num_classes = num_classes
        self._blocks = blocks

    @staticmethod
    def depthwise_block(x, stride, num_filters):
        outputs = DepthwiseConv2D(kernel_size=(3, 3), strides=stride, activation=None, padding="same")(x)
        outputs = BatchNormalization()(outputs)
        outputs = ReLU()(outputs)
        outputs = Conv2D(num_filters, kernel_size=(1, 1), activation=None, padding="same")(outputs)
        outputs = BatchNormalization()(outputs)
        outputs = ReLU()(outputs)
        return outputs

    def _build_graph(self):
        inputs = Input((32, 32, 3))
        outputs = Conv2D(32, kernel_size=(3, 3),strides=(2, 2), activation="relu", padding="valid")(inputs)
        outputs = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(outputs)
        for stride, num_filters in self._blocks:
            outputs = self.depthwise_block(outputs, stride=(stride, stride), num_filters=num_filters)

        outputs = Conv2D(self._num_classes, kernel_size=(1, 1), activation="relu", padding="valid")(outputs)
        outputs = GlobalAveragePooling2D()(outputs)
        outputs = softmax(outputs)
        return inputs, outputs

    def get(self):
        inputs, outputs = self._build_graph()
        self._model = keras.Model(inputs=inputs, outputs=outputs)
        return self._model


class MobileNetV2(object):
    def __init__(self, num_classes, blocks=[]):
        self._num_classes = num_classes
        self._blocks = blocks

    @staticmethod
    def _inverted_residual_block(x, stride, expansion_factor, output_channels):
        in_channels = x.shape[-1]
        expanded_channels = expansion_factor * in_channels

        out = Conv2D(expanded_channels, kernel_size=(1, 1), strides=1, activation=None, padding="same")(x)
        out = BatchNormalization()(out)
        out = ReLU()(out)
        out = DepthwiseConv2D(kernel_size=(3, 3), strides=stride, activation=None, padding="same")(out)
        out = BatchNormalization()(out)
        out = ReLU()(out)
        out = Conv2D(output_channels, kernel_size=(1, 1), strides=stride, activation=None, padding="same")(out)
        out = BatchNormalization()(out)
        if in_channels == output_channels and stride == 1:
            out = Add()([out, x])


        return out;


    def _build_graph(self):
        inputs = Input((32, 32, 3))
        outputs = Conv2D(32, kernel_size=(3, 3),strides=(1, 1), activation=None, padding="same")(inputs)
        outputs = BatchNormalization()(outputs)
        outputs = ReLU()(outputs)
        for stride, channels, expansion_factor, n in self._blocks:
            for _ in range(n):
                outputs = self._inverted_residual_block(outputs, stride=stride, expansion_factor=expansion_factor, output_channels=channels)

        outputs = Conv2D(256, kernel_size=(1, 1), activation=None, padding="same")(outputs)
        outputs = Dropout(0.25)(outputs)
        outputs = BatchNormalization()(outputs)
        outputs = ReLU()(outputs)
        outputs = GlobalAveragePooling2D()(outputs)
        outputs = Dense(self._num_classes, activation="sigmoid")(outputs)
        outputs = Dense(self._num_classes, activation="softmax")(outputs)
        return inputs, outputs

    def get(self):
        inputs, outputs = self._build_graph()
        self._model = keras.Model(inputs=inputs, outputs=outputs)
        return self._model

models = {
        "vgg_1": VGG(10, [(2, 32)]),
        "vgg_2": VGG(10, [(2, 32), (2, 64)]),
        "vgg_3": VGG(10, [(2, 32), (2, 64), (2, 128)]),
        "SqueezeNet_naive": SqueezeNet(10, [[(2, 16, 32)], [(4, 32, 64)]]),
        "SqueezeNet_naive_2": SqueezeNet(10, [[(2, 16, 32), (2, 32, 64)], [(2, 64, 128), (2, 64, 64)]]),
        "MobileNetV1_naive": MobileNetV1(10, [(1, 32), (1, 64), (1, 64), (1, 128), (2, 128), (1, 128)]),
        "MobileNetV2_naive": MobileNetV2(10, [(1, 16, 1, 1), (1, 24, 6, 3), (2, 32, 6, 1), (1, 64, 6, 3), (2, 128, 6, 1)]),
        }
