from abc import abstractmethod
import tensorflow as tf

class Model:

    def __init__(self):
        self._model = None
        self._session = None
        self._model_path = ""

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
    def build(self):
        return self

    def load_model(self):
        self._model = self.build()
        tf.train.Saver().restore(sess=self._session, save_path=self._model_path)
        return self

    def save_model(self):
        save_path = tf.train.Saver().save(sess=self._session, save_path=self._model_path)
        print("Model saved in path: %s" % save_path)
        return self
