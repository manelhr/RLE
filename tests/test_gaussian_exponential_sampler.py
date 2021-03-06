from rle.samplers.gaussian_exponential_sampler import GaussianExponentialSampler
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set_style("whitegrid")
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

fig, axs = plt.subplots(1, 3, figsize=(8, 2.5), sharex=True, sharey=True)

X, y = make_circles(noise=0.10, factor=0.8, n_samples=500)
decision = np.array([-0.42, 0.62])

# Plots original data
df = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])
df["Label"] = y
df_l0 = df[df.Label == 0]
df_l1 = df[df.Label == 1]
axs[0].plot(decision[0], decision[1], "b*", label="Decision")
axs[0].scatter(df_l0["Feature 1"].values, df_l0["Feature 2"].values, alpha=0.2)
axs[0].scatter(df_l1["Feature 1"].values, df_l1["Feature 2"].values, alpha=0.2)
axs[0].set_title("(a) Original Data (Train)")

# Plots classified data (Random Forest)
X_train, X_test, y_train, y_test = train_test_split(X, y)
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

df = pd.DataFrame(X_test, columns=["Feature 1", "Feature 2"])
df["Label"] = rf.predict(X_test)

# Plots gaussian sample
df_l0 = df[df.Label == 0]
df_l1 = df[df.Label == 1]
axs[1].plot(decision[0], decision[1], "b*", label="Decision")
axs[1].scatter(df_l0["Feature 1"].values, df_l0["Feature 2"].values, alpha=0.4)
axs[1].scatter(df_l1["Feature 1"].values, df_l1["Feature 2"].values, alpha=0.4)
axs[1].set_title("(b) Model Prediction (Test)")

# Plots Gaussian Sample around [-0.42, 0.62]
sampler = GaussianExponentialSampler(X, ["Feature 1, Feature 2"], ["numerical", "numerical"],
                                     y, "Label", "Categorical",
                                     100, 0.05,
                                     rf.predict_proba)

sample_f, sample_l, weights = sampler.sample(decision)

df2 = pd.DataFrame(sample_f, columns=["Feature 1", "Feature 2"])
df2["Label"] = sample_l

df_l0 = df2[df2.Label == 0]
df_l1 = df2[df2.Label == 1]
axs[2].plot(decision[0], decision[1], "b*", label="Decision")
axs[2].scatter(df_l0["Feature 1"].values, df_l0["Feature 2"].values, alpha=0.4, label='Class 1')
axs[2].scatter(df_l1["Feature 1"].values, df_l1["Feature 2"].values, alpha=0.4, label='Class 2')
axs[2].set_title("(c) Sampling: $[-0.42, 0.62]$")

axs[0].set_xlim(-1.5, 1.5)
axs[0].set_ylim(-1.5, 1.5)

# Calculates exponential kernel mesh
xk = np.arange(-1.5, 1.5, .025)
yk = np.arange(-1.5, 1.5, .025)
X, Y = np.meshgrid(xk, yk)
combined = np.stack([X.ravel(), Y.ravel()], axis=-1)
distances = pairwise_distances(combined, decision.reshape(1, -1), metric='euclidean')
exponential_distances = np.sqrt(np.exp(-(distances ** 2) / 2 ** 2))
Z = exponential_distances.reshape(len(X), len(X[0]))
cs = axs[2].contourf(X, Y, Z, alpha=0.25, cmap=plt.cm.bone)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.suptitle("Gaussian Sampling around the Neighborhood of a Decision")
plt.subplots_adjust(wspace=0.25, top=0.7)
plt.savefig("./imgs/gaussian_exponential_sampler.pdf", bbox_inches="tight")
