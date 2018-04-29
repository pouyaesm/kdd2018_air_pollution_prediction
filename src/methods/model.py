from abc import abstractmethod
import tensorflow as tf
from datetime import datetime

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

    def load_model(self, mode='final'):
        self._model = self.build()
        model_path = self._model_path.replace('#', mode)
        tf.train.Saver().restore(sess=self._session, save_path=model_path)
        return self

    def save_model(self, mode='final'):
        model_path = self._model_path.replace('#', mode)
        save_path = tf.train.Saver().save(sess=self._session, save_path=model_path)
        print(datetime.now().time(), "Model saved in path: %s" % save_path)
        return self
