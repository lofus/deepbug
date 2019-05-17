from keras.layers import Input, Dense, Activation, ZeroPadding1D, BatchNormalization, Flatten, Conv1D, Conv2D
from keras.layers import AveragePooling2D, ZeroPadding2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam


def deepbug_cnn_model(input_shape, num_output):
    """
    Implementation of the Deepbug model (cnn) with Keras FW.
    Arguments:
    input_shape -- shape of the images of the dataset
    num_output -- number of unique labels in the data
    Returns:
    model -- a Model() instance
    """
    print("deepbug_model_cnn start..")
    print("input_shape:", input_shape)
    print("num_output:", num_output)

    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)

    # CONV -> BN -> RELU
    X = Conv2D(16, (7, 7),  strides=(2, 2), name='conv0')(X)
    X = BatchNormalization(axis=2, name='bn0')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 4), name='max_pool')(X)

    # FLATTEN X
    X = Flatten()(X)

    # Softmax output
    X = Dense(num_output, activation='softmax', name='fc')(X)

    model = Model(inputs=X_input, outputs=X, name='DeepBugCNNModel')

    model.compile(
        loss="categorical_crossentropy", optimizer=Adam(lr=1e-4), metrics=["accuracy"]
    )

    return model
