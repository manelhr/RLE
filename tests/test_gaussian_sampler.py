from rle.samplers.gaussian_sampler import GaussianSampler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
import numpy as np
import pandas


X, y = make_circles(noise=0.10, factor=0.6, n_samples=500)
X_train, X_test, y_train, y_test = train_test_split(X, y)
decision = np.array([-0.3, 0.5])

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)


df = pandas.DataFrame(X, columns=["X Axis", "Y Axis"])

df["Label"] = y
ax = df[df.Label == 1].plot(kind="scatter", x="X Axis", y="Y Axis", color="DarkBlue")
df[df.Label == 0].plot(kind="scatter", x="X Axis", y="Y Axis", color="DarkGreen", ax=ax)
plt.plot(decision[0], decision[1], "*")

sampler = GaussianSampler(X, ["X Axis, Y Axis"], ["numerical", "numerical"],
                            y, "Label", "Categorical",
                            100, rf.predict_proba)

sample_f, sample_l = sampler.sample(decision)


df2 = pandas.DataFrame(sample_f, columns=["X Axis", "Y Axis"])
df2["Label"] = sample_l

df2[df2.Label == 1].plot(kind="scatter", x="X Axis", y="Y Axis", color="LightBlue", ax=ax)
df2[df2.Label == 0].plot(kind="scatter", x="X Axis", y="Y Axis", color="LightGreen", ax=ax)
plt.show()

