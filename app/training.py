from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

X, y = load_iris(return_X_y=True)

rf = RandomForestClassifier()
rf.fit(X, y)
print("X", X.shape)
print("y", y.shape)

joblib.dump(rf, r"f_model.joblib")
