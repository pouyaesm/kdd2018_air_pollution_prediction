from keras.models import Sequential
from keras.layers import Dense


class NeuralNet:

    @staticmethod
    def mlp(x_train, y_train, x_valid, y_valid):
        model = Sequential()
        input_size = x_train.shape[1]
        middle_size = int(input_size / 2)
        bottle_neck_size = int(input_size / 3)
        output_size = y_train.shape[1]
        model.add(Dense(units=input_size, input_dim=x_train.shape[1], activation='relu', kernel_initializer="uniform"))
        model.add(Dense(units=int(bottle_neck_size), activation='relu'))
        model.add(Dense(units=output_size))

        model.compile(loss='mean_absolute_error', optimizer='sgd')
        model.fit(x_train, y_train, epochs=50, batch_size=2000,
                  validation_data=(x_valid, y_valid), verbose=1)
        return model
