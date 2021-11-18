import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers.merge import add
from keras import regularizers
from keras.applications import ResNet50, VGG16

regular_constant=0


def resnet8(img_width, img_height, img_channels, output_dim):
    """
    Define model architecture.

    # Arguments
       img_width: Target image widht.
       img_height: Target image height.
       img_channels: Target image channels.
       output_dim: Dimension of model output.

    # Returns
       model: A Model instance.
    """

    # Input
    img_input = Input(shape=(img_height, img_width, img_channels))

    x1 = Conv2D(32, (5, 5), strides=[2,2], padding='same')(img_input)
    x1 = MaxPooling2D(pool_size=(3, 3), strides=[2,2])(x1)

    # First residual block
    x2 = keras.layers.normalization.BatchNormalization()(x1)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(32, (3, 3), strides=[2,2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-5))(x2)

    x2 = keras.layers.normalization.BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(32, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-5))(x2)

    x1 = Conv2D(32, (1, 1), strides=[2,2], padding='same')(x1)
    x3 = add([x1, x2])

    # Second residual block
    x4 = keras.layers.normalization.BatchNormalization()(x3)
    x4 = Activation('relu')(x4)
    x4 = Conv2D(64, (3, 3), strides=[2,2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x4)

    x4 = keras.layers.normalization.BatchNormalization()(x4)
    x4 = Activation('relu')(x4)
    x4 = Conv2D(64, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x4)

    x3 = Conv2D(64, (1, 1), strides=[2,2], padding='same')(x3)
    x5 = add([x3, x4])

    # Third residual block
    x6 = keras.layers.normalization.BatchNormalization()(x5)
    x6 = Activation('relu')(x6)
    x6 = Conv2D(128, (3, 3), strides=[2,2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x6)

    x6 = keras.layers.normalization.BatchNormalization()(x6)
    x6 = Activation('relu')(x6)
    x6 = Conv2D(128, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x6)

    x5 = Conv2D(128, (1, 1), strides=[2,2], padding='same')(x5)
    x7 = add([x5, x6])

    x = Flatten()(x7)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(output_dim)(x)

    # Define steering-collision model
    model = Model(inputs=[img_input], outputs=[x])
    #print(model.summary())

    return model



def resnet50(img_width, img_height, img_channels, output_dim):

    img_input = Input(shape=(img_height, img_width, img_channels))

    base_model = ResNet50(input_tensor=img_input,
                          weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    x = Dense(1024, activation='relu')(x)

    # Steering channel
    output = Dense(output_dim)(x)

    model = Model(inputs=[img_input], outputs=[output])
    #print(model.summary())

    return model


def resnet50_random_init(img_width, img_height, img_channels, output_dim):

    img_input = Input(shape=(img_height, img_width, img_channels))

    base_model = ResNet50(input_tensor=img_input,
                          weights=None, include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    x = Dense(1024, activation='relu')(x)

    # Steering channel
    output = Dense(output_dim)(x)

    model = Model(inputs=[img_input], outputs=[output])
    #print(model.summary())

    return model


def resnet18(img_width, img_height, img_channels, output_dim):
    """
    Define model architecture.

    # Arguments
       img_width: Target image widht.
       img_height: Target image height.
       img_channels: Target image channels.
       output_dim: Dimension of model output.

    # Returns
       model: A Model instance.
    """

    # Input
    img_input = Input(shape=(img_height, img_width, img_channels))

    x1 = Conv2D(32, (5, 5), strides=[2,2], padding='same')(img_input)
    x1 = MaxPooling2D(pool_size=(3, 3), strides=[2,2])(x1)

    # First residual block
    x2 = keras.layers.normalization.BatchNormalization()(x1)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(32, (3, 3), strides=[1,1], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(regular_constant))(x2)

    x2 = keras.layers.normalization.BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(32, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(regular_constant))(x2)

    x3 = add([x1, x2])

    # Second residual block
    x4 = keras.layers.normalization.BatchNormalization()(x3)
    x4 = Activation('relu')(x4)
    x4 = Conv2D(64, (3, 3), strides=[2,2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(regular_constant))(x4)

    x4 = keras.layers.normalization.BatchNormalization()(x4)
    x4 = Activation('relu')(x4)
    x4 = Conv2D(64, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(regular_constant))(x4)

    x3 = Conv2D(64, (1, 1), strides=[2,2], padding='same')(x3)
    x5 = add([x3, x4])

    # Third residual block
    x6 = keras.layers.normalization.BatchNormalization()(x5)
    x6 = Activation('relu')(x6)
    x6 = Conv2D(64, (3, 3), strides=[1,1], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(regular_constant))(x6)

    x6 = keras.layers.normalization.BatchNormalization()(x6)
    x6 = Activation('relu')(x6)
    x6 = Conv2D(64, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(regular_constant))(x6)

    x7 = add([x5, x6])

    # Fourth residual block
    x8 = keras.layers.normalization.BatchNormalization()(x7)
    x8 = Activation('relu')(x8)
    x8 = Conv2D(128, (3, 3), strides=[2,2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(regular_constant))(x8)

    x8 = keras.layers.normalization.BatchNormalization()(x8)
    x8 = Activation('relu')(x8)
    x8 = Conv2D(128, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(regular_constant))(x8)

    x7 = Conv2D(128, (1, 1), strides=[2,2], padding='same')(x7)
    x9 = add([x7, x8])

    # Fifth residual block
    x10 = keras.layers.normalization.BatchNormalization()(x9)
    x10 = Activation('relu')(x10)
    x10 = Conv2D(128, (3, 3), strides=[1,1], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(regular_constant))(x10)

    x10 = keras.layers.normalization.BatchNormalization()(x10)
    x10 = Activation('relu')(x10)
    x10 = Conv2D(128, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(regular_constant))(x10)

    x11 = add([x9, x10])

    # Sixth residual block
    x12 = keras.layers.normalization.BatchNormalization()(x11)
    x12 = Activation('relu')(x12)
    x12 = Conv2D(256, (3, 3), strides=[2,2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(regular_constant))(x12)

    x12 = keras.layers.normalization.BatchNormalization()(x12)
    x12 = Activation('relu')(x12)
    x12 = Conv2D(256, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(regular_constant))(x12)

    x11 = Conv2D(256, (1, 1), strides=[2,2], padding='same')(x11)
    x13 = add([x11, x12])

    # Seventh residual block
    x14 = keras.layers.normalization.BatchNormalization()(x13)
    x14 = Activation('relu')(x14)
    x14 = Conv2D(256, (3, 3), strides=[1,1], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(regular_constant))(x14)

    x14 = keras.layers.normalization.BatchNormalization()(x14)
    x14 = Activation('relu')(x14)
    x14 = Conv2D(256, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(regular_constant))(x14)

    x15 = add([x13, x14])

    # Eigth residual block
    x16 = keras.layers.normalization.BatchNormalization()(x15)
    x16 = Activation('relu')(x16)
    x16 = Conv2D(512, (3, 3), strides=[2,2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(regular_constant))(x16)

    x16 = keras.layers.normalization.BatchNormalization()(x16)
    x16 = Activation('relu')(x16)
    x16 = Conv2D(512, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(regular_constant))(x16)

    x15 = Conv2D(512, (1, 1), strides=[2,2], padding='same')(x15)
    x17 = add([x15, x16])

    x = GlobalAveragePooling2D()(x17)

    x = Dense(256)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(output_dim)(x)

    # Define steering-collision model
    model = Model(inputs=[img_input], outputs=[x])
    print(model.summary())

    return model


def nvidia_net(img_width, img_height, img_channels, output_dim):
    img_input = Input(shape=(img_height, img_width, img_channels))

    x = Conv2D(24, (5,5), strides=[2,2], padding='same')(img_input)
    x = Activation('relu')(x)

    x = Conv2D(36, (5,5), strides=[2,2], padding='same')(x)
    x = Activation('relu')(x)

    x = Conv2D(48, (5,5), strides=[2,2], padding='same')(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3,3), strides=[1,1], padding='same')(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3,3), strides=[1,1], padding='same')(x)
    x = Activation('relu')(x)

    x = Flatten()(x)
    x = Dense(100)(x)
    x = Dense(50)(x)
    x = Dense(10)(x)
    x = Dense(output_dim)(x)

    # Define steering-collision model
    model = Model(inputs=[img_input], outputs=[x])
    print(model.summary())

    return model


