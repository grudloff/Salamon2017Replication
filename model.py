from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Dropout, Activation, Flatten, Input, Reshape, 
                                    Conv2D, MaxPooling2D)
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2

num_labels = 10

def build_model(frames=128, bands=128, f_size = 5, channels = 0):
    
    model = Sequential()
    if channels == 0:
        # input shape : [samples, frames, bands]
        model.add(Input(shape=(frames, bands)))
        model.add(Reshape(target_shape=(frames,bands,1))) # add channel dim
    else:
        # input shape: [samples, frames, bands, channels]
        model.add(Input(shape=(frames, bands, channels)))

    # Layer 1 - 24 filters with a receptive field of (f,f), i.e. W has the
    # shape (24,f,f,1).  This is followed by (4,2) max-pooling over the last
    # two dimensions and a ReLU activation function
    model.add(Conv2D(24, f_size, padding='valid'))
    model.add(MaxPooling2D(pool_size=(4, 2)))
    model.add(Activation('relu'))

    # Layer 2 - 48 filters with a receptive field of (f,f), i.e. W has the 
    # shape (48, 24, f, f). Like L1 this is followed by (4,2) max-pooling 
    # and a ReLU activation function.
    model.add(Conv2D(48, f_size, padding='valid'))
    model.add(MaxPooling2D(pool_size=(4, 2)))
    model.add(Activation('relu'))     

    # Layer 3 - 48 filters with a receptive field of (f,f), i.e. W has the
    # shape (48, 48, f, f). This is followed by a ReLU but no pooling.
    model.add(Conv2D(48, f_size, padding='valid'))
    model.add(Activation('relu'))

    # flatten output into a single dimension
    model.add(Flatten())
    model.add(Dropout(0.5))

    # Layer 4 - a fully connected NN layer of 64 hidden units, L2 penalty of 0.001
    model.add(Dense(64, kernel_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # Layer 5 - an output layer with one output unit per class, with L2 penalty, 
    # followed by a softmax activation function
    model.add(Dense(num_labels, kernel_regularizer=l2(0.001)))
    model.add(Activation('softmax'))

    # compile model
    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer=SGD(lr=0.01))

    return model

def build_model_multi(frames=128, bands=128, channels=0, f_size=5):
    # variation of the previous model to allow to evaluate over a set of adjacent samples
    # and average the output like it is done on the original paper

    model = Sequential()
    if channels == 0:
        # input shape : [samples, frames, bands]
        model.add(Input(shape=(frames, bands)))
        model.add(Reshape(target_shape=(frames,bands,1))) # add channel dim
    else:
        # input shape: [samples, frames, bands, channels]
        model.add(Input(shape=(frames, bands, channels)))

    # Layer 1 - 24 filters with a receptive field of (f,f), i.e. W has the
    # shape (24,f,f,1).  This is followed by (4,2) max-pooling over the last
    # two dimensions and a ReLU activation function
    model.add(TimeDistributed(Conv2D(24, f_size, padding='valid')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(4, 2))))
    model.add(TimeDistributed(Activation('relu')))

    # Layer 2 - 48 filters with a receptive field of (f,f), i.e. W has the 
    # shape (48, 24, f, f). Like L1 this is followed by (4,2) max-pooling 
    # and a ReLU activation function.
    model.add(TimeDistributed(Conv2D(48, f_size, padding='valid')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(4, 2))))
    model.add(TimeDistributed(Activation('relu')))     

    # Layer 3 - 48 filters with a receptive field of (f,f), i.e. W has the
    # shape (48, 48, f, f). This is followed by a ReLU but no pooling.
    model.add(TimeDistributed(Conv2D(48, f_size, padding='valid')))
    model.add(TimeDistributed(Activation('relu')))

    # flatten output into a single dimension
    model.add(TimeDistributed(Flatten()))
    model.add(Dropout(0.5))

    # Layer 4 - a fully connected NN layer of 64 hidden units, L2 penalty of 0.001
    model.add(TimeDistributed(Dense(64, kernel_regularizer=l2(0.001))))
    model.add(TimeDistributed(Activation('relu')))
    model.add(Dropout(0.5))

    # Layer 5 - an output layer with one output unit per class, with L2 penalty, 
    # followed by a softmax activation function
    model.add(TimeDistributed(Dense(num_labels, kernel_regularizer=l2(0.001))))
    model.add(TimeDistributed((Activation('softmax'))))
    model.add(GlobalAveragePooling1D()) #average across time

    # compile model
    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer=SGD(lr=0.01))

    return model
