from abc import ABC, abstractmethod
class Model:

    @abstractmethod
    def train(self):
        return self

    @abstractmethod
    def predict(self, x):
        return x

    @abstractmethod
    def evaluate(self, actual, forecast):
        return 0

    @abstractmethod
    def save_model(self):
        return self

    @abstractmethod
    def load_model(self):
        return self
