import keras
from keras.layers import Input, Dense, Activation, ZeroPadding1D, BatchNormalization, Flatten, Conv1D, Conv2D, Dropout
from keras.layers import AveragePooling2D, ZeroPadding2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras import regularizers
from keras.models import Model, Sequential
from keras.optimizers import Adam

def deepbug_cnn_model(input_shape, num_output):
    """
    Implementation of the Deepbug model (train for cnn path) with Keras FW.
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

    # 1st Convolutional Layer
    # CONV -> BN -> RELU -> AveragePooling
    X = BatchNormalization(axis=2, name='bn0')(X)
    X = Conv2D(16, (5, 5),  strides=(1, 1), name='conv0')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name='max_pool0')(X)
    X = Dropout(0.4)(X)

    # 2nd Convolutional Layer
    X = Conv2D(32, (3, 3), strides=(1, 1), name='conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name='max_pool1')(X)
    X = Dropout(0.4)(X)

    # 3rd Convolutional Layer
    X = Conv2D(64, (3, 3), strides=(1, 1), name='conv2')(X)
    X = Activation('relu')(X)
    X = AveragePooling2D((2, 2), name='average_pool2')(X)
    X = Dropout(0.4)(X)

    # FLATTEN X
    X = Flatten()(X)

    X = Dense(500, activation='relu', name='fc0')(X)

    # Softmax output
    X = Dense(num_output, activation='softmax', name='fc1')(X)

    model = Model(inputs=X_input, outputs=X, name='DeepBugCNNModel')

    model.compile(
        loss="categorical_crossentropy", optimizer=Adam(lr=1e-4), metrics=["accuracy"]
    )

    return model

def deepbug_alexnet_model(input_shape, num_output):
    """
    Prototype for Deepbug model (train for cnn path) with Keras FW.
    Arguments:
    input_shape -- shape of the images of the dataset
    num_output -- number of unique labels in the data
    Returns:
    model -- a Model() instance
    """
    model = Sequential()

    # normalize input
    model.add(BatchNormalization(axis=2))

    # 1st Convolutional Layer
    model.add(Conv2D(filters=96, input_shape=input_shape, kernel_size=(11,11), strides=(2,2), padding='valid'))
    model.add(Activation('relu'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

    # 3rd Convolutional Layer
    model.add(Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # 4th Convolutional Layer
    #model.add(BatchNormalization(axis=2))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    # 5th Convolutional Layer
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # Passing it to a Fully Connected layer
    model.add(Flatten())

    """
    # 1st Fully Connected Layer
    model.add(Dense(4096, input_shape=(input_shape,)))
    model.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.4))

    # 2nd Fully Connected Layer
    model.add(Dense(4096))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))
    """

    # 3rd Fully Connected Layer
    #model.add(Dense(2000, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(2000, activation='relu'))
    # Add Dropout
    model.add(Dropout(0.4))

    # Output Layer
    model.add(Dense(num_output))
    model.add(Activation('softmax'))

    # Compile the model
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=["accuracy"])

    return model


def deepbug_rnn_model(input_shape, num_output, num_rnn_unit=512, num_dense_unit=1000):
    """
    Implementation of the Deepbug model (rnn path) with Keras FW.
    Arguments:
    input_shape -- shape of the images of the dataset
    num_output -- number of unique labels in the data
    Returns:
    model -- a Model() instance
    """
    input_1 = Input(shape=input_shape, dtype="float32")

    forwards_1 = GRU(num_rnn_unit, return_sequences=True, dropout=0.2)(input_1)
    attention_1 = Dense(1, activation="tanh")(forwards_1)
    attention_1 = Flatten()(attention_1)  # squeeze (None,50,1)->(None,50)
    attention_1 = Activation("softmax")(attention_1)
    attention_1 = RepeatVector(num_rnn_unit)(attention_1)
    attention_1 = Permute([2, 1])(attention_1)
    attention_1 = multiply([forwards_1, attention_1])
    attention_1 = Lambda(lambda xin: K.sum(xin, axis=1), output_shape=(num_rnn_unit,))(
        attention_1
    )

    last_out_1 = Lambda(lambda xin: xin[:, -1, :])(forwards_1)
    sent_representation_1 = concatenate([last_out_1, attention_1])

    after_dp_forward_5 = BatchNormalization()(sent_representation_1)
    backwards_1 = GRU(
        num_rnn_unit, return_sequences=True, dropout=0.2, go_backwards=True
    )(input_1)

    attention_2 = Dense(1, activation="tanh")(backwards_1)
    attention_2 = Flatten()(attention_2)
    attention_2 = Activation("softmax")(attention_2)
    attention_2 = RepeatVector(num_rnn_unit)(attention_2)
    attention_2 = Permute([2, 1])(attention_2)
    attention_2 = multiply([backwards_1, attention_2])
    attention_2 = Lambda(lambda xin: K.sum(xin, axis=1), output_shape=(num_rnn_unit,))(
        attention_2
    )

    last_out_2 = Lambda(lambda xin: xin[:, -1, :])(backwards_1)
    sent_representation_2 = concatenate([last_out_2, attention_2])

    after_dp_backward_5 = BatchNormalization()(sent_representation_2)

    merged = concatenate([after_dp_forward_5, after_dp_backward_5])
    after_merge = Dense(num_dense_unit, activation="relu")(merged)
    after_dp = Dropout(0.4)(after_merge)
    output = Dense(num_output, activation="softmax")(after_dp)
    model = Model(input=input_1, output=output)
    model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=1e-4), metrics=["accuracy"])

    return model