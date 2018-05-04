from abc import abstractmethod
import tensorflow as tf
from datetime import datetime

class Model:

    def __init__(self):
        self.model = None
        self.session = None
        self.model_path = ""
        self.graph = None
        self.scope = ""  # graph variables scope to be separated from other graphs

    def build(self, session: tf.Session=None):
        # create tensor-flow session and graph
        self.graph = tf.Graph() if session is None else session.graph
        with self.graph.as_default():
            self.session = tf.Session() if session is None else session
            self.model = self.build_model()
        return self

    def load_model(self, mode='final', session: tf.Session=None, model_scope=''):
        """
        :param model_scope:
        :param session:
        :param mode:
        :return:
        """
        # create the model and load the model data (variables) into it
        with tf.variable_scope(model_scope):
            self.build(session)
        with self.graph.as_default():
            model_path = self.model_path.replace('#', mode)
            # saver = tf.train.import_meta_graph(model_path + '.meta')
            # Add loaded graph to model's session
            # Remove model_scope from variable names to match the scope-free saved names
            var_list = {v.name.lstrip("%s/" % model_scope)[:-2]: v
                             for v in tf.global_variables() if v.name.startswith(model_scope)}
            tf.train.Saver(var_list=var_list).restore(sess=self.session, save_path=model_path)
        # self.graph = tf.Graph()
        # with self.graph.as_default():
        #     model_path = self.model_path.replace('#', mode)
        #     # saver = tf.train.import_meta_graph(model_path + '.meta')
        #     # add loaded graph to model's session
        #     self.session = tf.Session()
        #     saver.restore(sess=self.session, save_path=model_path)
        return self

    def save_model(self, mode='final'):
        with self.graph.as_default():
            model_path = self.model_path.replace('#', mode)
            save_path = tf.train.Saver().save(sess=self.session, save_path=model_path)
            print(datetime.now().time(), "Model saved in path: %s" % save_path)
        return self

    def build_model(self):
        """
        :return:
        :rtype: tensorflow.Tensor
        """
        return None

    def train(self):
        with self.graph.as_default():
            self.train_model()
        return self

    def train_model(self):
        return self

    @abstractmethod
    def predict(self, x):
        return x

    @abstractmethod
    def evaluate(self, actual, forecast):
        return 0

    def get_graph(self):
        """
        :return:
        :rtype: tensorflow.Graph
        """
        return self.graph

    def get_model(self):
        """
        :return:
        :rtype: dict
        """
        return self.model
