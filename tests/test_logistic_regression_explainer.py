from rle.explainers.logistic_regression_explainer import LogisticRegressionExplainer
from rle.samplers.gaussian_sampler import GaussianSampler
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
import numpy as np
import pandas

fig, ax = plt.subplots(1, 1, figsize=(7, 6), sharex=True, sharey=True)

# Initializes dataset and variables
X, y = make_circles(noise=0.10, factor=0.6, n_samples=1000)
decision = np.array([-0.42, 0.62])
measure = 0.5

# Initializes model
X_train, X_test, y_train, y_test = train_test_split(X, y)
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

# Initializes sampler
sampler = GaussianSampler(X, ["Feature 1", "Feature 2"], ["numerical", "numerical"],
                          y, "Label", "Categorical",
                          100, rf.predict_proba)

# Initializes explainer
explainer = LogisticRegressionExplainer(sampler, measure)

# Plots decision
ax.plot(decision[0], decision[1], "b*", label="Decision")

# Performs Logistic Regression
weights = explainer.explain(decision)
last_decision, df = explainer.explanation_data()

# Plots gaussian sample
df[df.Label == 1].plot(kind="scatter", x="Feature 1", y="Feature 2", color="Blue", ax=ax)
df[df.Label == 0].plot(kind="scatter", x="Feature 1", y="Feature 2", color="Green", ax=ax)

# Plots regression line
xs = [min(df["Feature 1"].values), max(df["Feature 1"].values)]
ys = list(map(lambda x: (-weights[0][1] * x - weights[2][1]) / weights[1][1], xs))
ax.plot(xs, ys)

# Plots exponential kernel
xk = np.arange(min(df["Feature 1"].values), max(df["Feature 1"].values), .025)
yk = np.arange(min(df["Feature 2"].values), max(df["Feature 2"].values), .025)
X, Y = np.meshgrid(xk, yk)
combined = np.stack([X.ravel(), Y.ravel()], axis=-1)
distances = pairwise_distances(combined, decision.reshape(1, -1), metric='euclidean')
exponential_distances = np.sqrt(np.exp(-(distances ** 2) / measure ** 2))
Z = exponential_distances.reshape(len(X), len(X[0]))
cs = ax.contourf(X, Y, Z, alpha=0.3)

ax.set_xlim(min(df["Feature 1"].values) + 0.010, max(df["Feature 1"].values) - 0.010)
ax.set_ylim(min(df["Feature 2"].values) + 0.010, max(df["Feature 2"].values) - 0.010)
ax.set_title("Weighted Logistic Regression")
plt.savefig("./imgs/logistic_regression_explainer.pdf", bbox_inches="tight")
