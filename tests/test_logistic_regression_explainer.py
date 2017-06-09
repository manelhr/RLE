from rle.samplers.gaussian_sampler import GaussianSampler
from rle.explainers.logistic_regression_explainer import LogisticRegressionExplainer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_circles
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas

fig, ax = plt.subplots(1, 1, figsize=(7, 6), sharex=True, sharey=True)

X, y = make_circles(noise=0.10, factor=0.6, n_samples=1000)

# Trains model

X_train, X_test, y_train, y_test = train_test_split(X, y)
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
decision = np.array([-0.42, 0.62])
measure = 0.5

# Plots Gaussian Sample around [-0.42, 0.62]

sampler = GaussianSampler(X, ["Feature 1, Feature 2"], ["numerical", "numerical"],
                          y, "Label", "Categorical",
                          100, rf.predict_proba)

LogisticRegressionExplainer(sampler, measure)

# sample_f, sample_l = sampler.sample(decision)
#
# df = pandas.DataFrame(sample_f, columns=["Feature 1", "Feature 2"])
# df["Label"] = sample_l
#
# df[df.Label == 1].plot(kind="scatter", x="Feature 1", y="Feature 2", color="LightBlue", ax=ax)
# df[df.Label == 0].plot(kind="scatter", x="Feature 1", y="Feature 2", color="LightGreen", ax=ax)
# ax.plot(decision[0], decision[1], "b*", label="Decision")
# ax.set_title("Weighted Logistic Regression")
#
# # Performs Logistic Regression
#
# distances = pairwise_distances(sample_f, decision.reshape(1, -1), metric='euclidean')
# distances_ravel = distances.ravel()
# exponential_distances = np.sqrt(np.exp(-(distances_ravel ** 2) / measure ** 2))
#
# print(distances_ravel, exponential_distances)
#
# model = LogisticRegression()
# model.fit(sample_f, sample_l, exponential_distances)
# xs = [min(df["Feature 1"].values), max(df["Feature 1"].values)]
# ys = list(map(lambda x: (-model.coef_[0][0] * x - model.intercept_[0]) / model.coef_[0][1], xs))
#
# ax.plot(xs, ys)
#
# xk = np.arange(min(df["Feature 1"].values), max(df["Feature 1"].values), .025)
# yk = np.arange(min(df["Feature 2"].values), max(df["Feature 2"].values), .025)
# kdist = pairwise_distances(sample_f, decision.reshape(1, -1), metric='euclidean')
# zk = np.sqrt(np.exp(-(kdist ** 2) / measure ** 2))
#
#
# X, Y = np.meshgrid(xk, yk)
# combined = np.stack([X.ravel(), Y.ravel()], axis=-1)
# distances = pairwise_distances(combined, decision.reshape(1, -1), metric='euclidean')
# exponential_distances = np.sqrt(np.exp(-(distances ** 2) / measure ** 2))
#
# Z = exponential_distances.reshape(len(X), len(X[0]))
#
# cs = ax.contour(X, Y, Z, alpha=0.3)
#
# plt.clabel(cs, inline=1, fontsize=10)
#
# plt.savefig("./imgs/logistic_regression_explainer.pdf", bbox_inches="tight")
