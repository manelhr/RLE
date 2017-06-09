from rle.samplers.gaussian_sampler import GaussianSampler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
import numpy as np
import pandas

fig, ax = plt.subplots(3, 1, figsize=(7, 6), sharex=True, sharey=True)

X, y = make_circles(noise=0.10, factor=0.6, n_samples=1000)

# Trains model

X_train, X_test, y_train, y_test = train_test_split(X, y)
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
decision = np.array([-0.42, 0.62])

# Plots Gaussian Sample around [-0.42, 0.62]

sampler = GaussianSampler(X, ["Feature 1, Feature 2"], ["numerical", "numerical"],
                          y, "Label", "Categorical",
                          100, rf.predict_proba)

sample_f, sample_l = sampler.sample(decision)

df2 = pandas.DataFrame(sample_f, columns=["Feature 1", "Feature 2"])
df2["Label"] = sample_l

df2[df2.Label == 1].plot(kind="scatter", x="Feature 1", y="Feature 2", color="LightBlue", ax=ax[0])
df2[df2.Label == 0].plot(kind="scatter", x="Feature 1", y="Feature 2", color="LightGreen", ax=ax[0])
ax[0].plot(decision[0], decision[1], "b*", label="Decision")
ax[0].set_title("Gaussian Sampling Around [-0.42, 0.62]")

plt.show()