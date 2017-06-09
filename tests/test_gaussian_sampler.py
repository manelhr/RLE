from rle.samplers.gaussian_sampler import GaussianSampler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
import numpy as np
import pandas

fig, ax = plt.subplots(3, 1, figsize=(7, 6), sharex=True, sharey=True)

X, y = make_circles(noise=0.10, factor=0.6, n_samples=1000)
decision = np.array([-0.42, 0.62])

# Plots original data

df = pandas.DataFrame(X, columns=["Feature 1", "Feature 2"])
df["Label"] = y

df[df.Label == 1].plot(kind="scatter", x="Feature 1", y="Feature 2", color="LightBlue", alpha=0.8, ax=ax[0])
df[df.Label == 0].plot(kind="scatter", x="Feature 1", y="Feature 2", color="LightGreen", alpha=0.8, ax=ax[0])
ax[0].set_title("Original Data (Train)")

# Plots classified data (Random Forest)

X_train, X_test, y_train, y_test = train_test_split(X, y)
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

df = pandas.DataFrame(X_test, columns=["Feature 1", "Feature 2"])
df["Label"] = rf.predict(X_test)

ax[1].plot(decision[0], decision[1], "b*", label="Decision")
df[df.Label == 1].plot(kind="scatter", x="Feature 1", y="Feature 2", color="LightBlue", alpha=0.8, ax=ax[1],style="+")
df[df.Label == 0].plot(kind="scatter", x="Feature 1", y="Feature 2", color="LightGreen", alpha=0.8, ax=ax[1], style="-")
ax[1].set_title("Model Prediction (Test)")

# Plots Gaussian Sample around [-0.42, 0.62]

sampler = GaussianSampler(X, ["Feature 1, Feature 2"], ["numerical", "numerical"],
                          y, "Label", "Categorical",
                          100, rf.predict_proba)

sample_f, sample_l = sampler.sample(decision)

df2 = pandas.DataFrame(sample_f, columns=["Feature 1", "Feature 2"])
df2["Label"] = sample_l

df2[df2.Label == 1].plot(kind="scatter", x="Feature 1", y="Feature 2", color="LightBlue", ax=ax[2])
df2[df2.Label == 0].plot(kind="scatter", x="Feature 1", y="Feature 2", color="LightGreen", ax=ax[2])
ax[2].plot(decision[0], decision[1], "b*", label="Decision")
ax[2].set_title("Gaussian Sampling Around [-0.42, 0.62]")

plt.legend()
plt.savefig("./imgs/gaussian_sampler.pdf", bbox_inches="tight")
