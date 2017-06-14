from rle.samplers.gaussian_sampler import GaussianSampler
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

fig, axs = plt.subplots(1, 3, figsize=(8, 2), sharex=True, sharey=True)

X, y = make_circles(noise=0.10, factor=0.8, n_samples=500)
decision = np.array([-0.42, 0.62])

# Plots original data

df = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])
df["Label"] = y

df[df.Label == 1].plot(kind="scatter", x="Feature 1", y="Feature 2", color="Blue", alpha=0.8, ax=axs[0])
df[df.Label == 0].plot(kind="scatter", x="Feature 1", y="Feature 2", color="Green", alpha=0.8, ax=axs[0])
axs[0].set_title("Original Data (Train)")

# Plots classified data (Random Forest)

X_train, X_test, y_train, y_test = train_test_split(X, y)
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

df = pd.DataFrame(X_test, columns=["Feature 1", "Feature 2"])
df["Label"] = rf.predict(X_test)

axs[1].plot(decision[0], decision[1], "b*", label="Decision")
df[df.Label == 1].plot(kind="scatter", x="Feature 1", y="Feature 2", color="LightBlue", alpha=0.8, ax=axs[1], style="+")
df[df.Label == 0].plot(kind="scatter", x="Feature 1", y="Feature 2", color="LightGreen", alpha=0.8, ax=axs[1],
                       style="-")
axs[1].set_title("Model Prediction (Test)")

# Plots Gaussian Sample around [-0.42, 0.62]

sampler = GaussianSampler(X, ["Feature 1, Feature 2"], ["numerical", "numerical"],
                          y, "Label", "Categorical",
                          100, rf.predict_proba)

sample_f, sample_l = sampler.sample(decision)

df2 = pd.DataFrame(sample_f, columns=["Feature 1", "Feature 2"])
df2["Label"] = sample_l

c1 = df2[df2.Label == 1].plot(kind="scatter", x="Feature 1", y="Feature 2",
                              color="LightBlue", ax=axs[2], label='Class 1')
c2 = df2[df2.Label == 0].plot(kind="scatter", x="Feature 1", y="Feature 2",
                              color="LightGreen", ax=axs[2], label='Class 2')
c3 = axs[2].plot(decision[0], decision[1], "b*", label="Decision")
axs[2].set_title("Sampling: $[-0.42, 0.62]$")

axs[0].set_xlim(-1.5, 1.5)
axs[0].set_ylim(-1.5, 1.5)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.suptitle("Gaussian Sampling around the Neighborhood of a Decision")
plt.subplots_adjust(wspace=0.25, top=0.7)
plt.savefig("./imgs/gaussian_sampler.pdf", bbox_inches="tight")
