from rle.explainers.logistic_regression_explainer import LogisticRegressionExplainer
from rle.samplers.gaussian_exponential_sampler import GaussianExponentialSampler
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import pandas as pd
import numpy as np


def same_side(p1, p2, a, b):
    cp1 = np.cross(b - a, p1 - a)
    cp2 = np.cross(b - a, p2 - a)
    if np.inner(cp1, cp2) >= 0:
        return True
    else:
        return False


def inside_triangle(p, a, b, c):
    if same_side(p, a, b, c) and same_side(p, b, a, c) and same_side(p, c, a, b):
        return True
    else:
        return False

fig, axs = plt.subplots(1, 3, figsize=(8, 2.5), sharex=True, sharey=True)

# Initializes double triangle dataset
X = np.random.rand(5000, 2)
triangle = np.array([[0.4, 0.4], [0.6, 0.6], [0.2, 0.8]])
decision = np.array([0.35, 0.55])

y = np.array(list(map(lambda x: int(x[1] < x[0] or inside_triangle(x, triangle[0], triangle[1], triangle[2])), X)))

df = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])
df["Label"] = y

df_l0 = df[df.Label == 0]
df_l1 = df[df.Label == 1]

# Plots classified data (Random Forest)
X_train, X_test, y_train, y_test = train_test_split(X, y)
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

df = pd.DataFrame(X_test, columns=["Feature 1", "Feature 2"])
df["Label"] = rf.predict(X_test)

# Plots gaussian sample
df_l0 = df[df.Label == 0]
df_l1 = df[df.Label == 1]
axs[0].plot(decision[0], decision[1], "b*", label="Decision")
axs[0].scatter(df_l0["Feature 1"].values, df_l0["Feature 2"].values, alpha=0.4)
axs[0].scatter(df_l1["Feature 1"].values, df_l1["Feature 2"].values, alpha=0.4)
axs[0].set_title("(a) Model Prediction")


plt.show()
