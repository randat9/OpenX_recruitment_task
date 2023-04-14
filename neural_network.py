import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def build_model(input_shape, num_classes, num_layers=2, num_units=64, dropout_rate=0.2, learning_rate=0.001):
    """
    Builds a neural network model.

    Args:
    - input_shape: a tuple representing the shape of the input data
    - num_classes: an integer representing the number of classes
    - num_layers: an integer representing the number of hidden layers
    - num_units: an integer representing the number of units per hidden layer
    - dropout_rate: a float representing the dropout rate
    - learning_rate: a float representing the learning rate

    Returns:
    - a compiled neural network model
    """

    # Define the model architecture
    model = Sequential()
    model.add(Dense(num_units, activation='relu', input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    for i in range(num_layers-1):
        model.add(Dense(num_units, activation='relu'))
        model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model
