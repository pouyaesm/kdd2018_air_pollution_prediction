from gcforest.gcforest import GCForest
from src.gcforest import util
import numpy as np

X_train = np.array([[0], [1], [2], [3], [4], [5], [6], [7]])
y_train = np.array([0, -1, 0, -1, 0, -1, 0, -1])
print(X_train.shape)
print(y_train.shape)
X_test = np.array([[8], [9], [10], [11]])
gc = GCForest(util.get_toy_config()) # should be a dict
X_train_enc = gc.fit_transform(X_train, y_train)
y_pred = gc.predict(X_test)


