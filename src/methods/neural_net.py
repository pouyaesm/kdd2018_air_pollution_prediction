from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Activation
from keras.optimizers import Nadam, Adam


class NeuralNet:

    @staticmethod
    def add_default_layers(model):
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        return model

    @staticmethod
    def mlp(x_train, y_train, x_valid, y_valid, loss='mean_absolute_error'):
        model = Sequential()
        input_size = x_train.shape[1]
        middle_size = int(input_size / 2)
        bottle_neck_size = int(input_size / 3)
        output_size = y_train.shape[1]
        model.add(Dense(units=input_size, input_dim=input_size))
        NeuralNet.add_default_layers(model)
        # model.add(Dense(units=int(middle_size), kernel_initializer="uniform"))
        # NeuralNet.add_default_layers(model)
        # model.add(Dense(units=int(middle_size), kernel_initializer="uniform"))
        # NeuralNet.add_default_layers(model)
        model.add(Dense(units=int(middle_size)))
        NeuralNet.add_default_layers(model)
        # model.add(Dense(units=int(bottle_neck_size)))
        # NeuralNet.add_default_layers(model)
        # model.add(Dense(units=int(middle_size)))
        # NeuralNet.add_default_layers(model)
        model.add(Dense(units=output_size))

        model.compile(loss=loss, optimizer=Adam(lr=0.01))
        # model.fit(x_train, y_train, epochs=75, batch_size=5000,
        #           validation_data=(x_valid, y_valid), verbose=1)
        model.fit(x_train, y_train, epochs=100, batch_size=1000, verbose=1)
        return model

