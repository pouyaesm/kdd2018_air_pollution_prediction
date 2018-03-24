# a toy configuration for gcForest copied from the example of the library

from gcforest.gcforest import GCForest
import numpy as np

def get_toy_config():
    config = {}
    ca_config = {}
    ca_config["random_state"] = 0
    ca_config["max_layers"] = 100
    ca_config["early_stopping_rounds"] = 3
    ca_config["n_classes"] = 3  # some where down the execution path, no. class must be larger than 2!
    ca_config["estimators"] = []
    ca_config["estimators"].append(
            {"n_folds": 2, "type": "XGBClassifier", "n_estimators": 10, "max_depth": 5,
             "objective": "multi:softprob", "silent": True, "nthread": -1, "learning_rate": 0.1} )
    ca_config["estimators"].append({"n_folds": 2, "type": "RandomForestClassifier", "n_estimators": 10, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 2, "type": "ExtraTreesClassifier", "n_estimators": 10, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 2, "type": "LogisticRegression"})
    config["cascade"] = ca_config
    return config


if __name__ == "__main__":
    X_train = np.array([[0], [1], [2], [3], [4], [5], [6], [7]])
    y_train = np.array([0, 0, 0, 1, 1, 1, 2, 2])
    X_test = np.array([[-2], [-1], [2.5], [3.5], [8], [10]])
    y_test = np.array([0, 0, 1, 1, 2, 2])
    gc = GCForest(get_toy_config()) # should be a dict
    X_train_enc = gc.fit_transform(X_train, y_train)
    y_pred = gc.predict(X_test)

    # evaluate the model
    print(y_test)
    print(y_pred)
    acc = (y_test == y_pred).sum() / len(y_test)
    print("Test Accuracy of GcForest = {:.2f} %".format(acc * 100))
