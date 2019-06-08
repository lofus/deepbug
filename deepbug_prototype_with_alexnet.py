import keras
from matplotlib import pyplot as plt
from keras.datasets import mnist
from keras.datasets import cifar10
from keras.datasets import fashion_mnist
from keras.layers import Input, Dense, Activation, ZeroPadding1D, BatchNormalization, Flatten, Conv1D, Conv2D, Dropout
from keras.layers import AveragePooling2D, ZeroPadding2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras import regularizers
from keras.models import Model, Sequential
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from plot_loss import PlotLosses, plot_3d_matrice


def alexnet_model(input_shape, output_dense):

    """
    Adapt from AlexNet, for DeepBug prototype and benchmark
    """
    model = Sequential()

    #normalize input
    model.add(BatchNormalization(axis=2))
    """
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
    """

    # 3rd Convolutional Layer
    model.add(Conv2D(filters=16, kernel_size=(5,5), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # 4th Convolutional Layer
    model.add(BatchNormalization(axis=2))
    model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    # 5th Convolutional Layer
    model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

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
    model.add(Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    # Add Dropout
    model.add(Dropout(0.4))

    # Output Layer
    model.add(Dense(output_dense))
    model.add(Activation('softmax'))

    # Compile the model
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=["accuracy"])

    return model

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

    # 1st Convolutional Layer
    # CONV -> BN -> RELU -> AveragePooling
    X = BatchNormalization(axis=2, name='bn0')(X)
    X = Conv2D(16, (4, 4),  strides=(1, 1), name='conv0')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name='max_pool0')(X)
    X = Dropout(0.2)(X)

    # 2nd Convolutional Layer
    X = Conv2D(32, (3, 3), strides=(1, 1), name='conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name='max_pool1')(X)
    X = Dropout(0.2)(X)

    # 3rd Convolutional Layer
    X = Conv2D(64, (3, 3), strides=(1, 1), name='conv2')(X)
    X = Activation('relu')(X)
    #X = MaxPooling2D((2, 2), name='max_pool2')(X)
    X = Dropout(0.2)(X)

    # 4th Convolutional Layer
    X = Conv2D(128, (3, 3), strides=(1, 1), name='conv3')(X)
    X = Activation('relu')(X)
    X = Dropout(0.2)(X)

    # FLATTEN X
    X = Flatten()(X)

    X = Dense(100, activation='relu', name='fc0')(X)

    # Softmax output
    X = Dense(num_output, activation='softmax', name='softmax')(X)

    model = Model(inputs=X_input, outputs=X, name='DeepBugCNNModel')

    model.compile(
        loss="categorical_crossentropy", optimizer=Adam(lr=1e-4), metrics=["accuracy"]
    )

    return model

print("Load Keras cifar10")
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print("preprocess dataset")
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print("x_train.shape", x_train.shape)
print("y_train.shape", y_train.shape)
print("x_test.shape", x_test.shape)
print("y_test.shape", y_test.shape)

# create the model
model = deepbug_cnn_model(x_train[0].shape, y_train.shape[1])

# for plot losses
plot_losses = PlotLosses()
checkpt = ModelCheckpoint('deepbug.model.h5', monitor='val_acc', save_best_only=True,save_weights_only=True)
early_stopping = EarlyStopping(monitor="val_loss", patience=8)

print("model summary:")
model.summary()

hist = model.fit(
    x_train,
    y_train,
    epochs=100,
    validation_split=0.1,
    batch_size=2048,
    callbacks=[plot_losses, checkpt, early_stopping]
)

print("model summary:")
model.summary()

prediction = model.predict(x_test)
print(prediction)

