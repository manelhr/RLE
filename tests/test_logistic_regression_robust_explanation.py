from rle.explainers.logistic_regression_explainer import LogisticRegressionExplainer
from rle.samplers.gaussian_exponential_sampler import GaussianExponentialSampler
from rle.depicters.depicter_bar_weights import DepicterBarWeights
from sklearn.model_selection import train_test_split
from rle.explanation.explanation import Explanation
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set_style("whitegrid")
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


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


fig, axs = plt.subplots(1, 3, figsize=(8, 2.5))

# Initializes double triangle dataset
X = np.random.rand(5000, 2)
triangle = np.array([[0.4, 0.4], [0.6, 0.6], [0.01, 0.99]])
decision = np.array([0.27, 0.64])
y = np.array(list(map(lambda x: int(x[1] < x[0] or inside_triangle(x, triangle[0], triangle[1], triangle[2])), X)))

# Plots classified data (Random Forest)
X_train, X_test, y_train, y_test = train_test_split(X, y)
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
df = pd.DataFrame(X_test, columns=["F1", "F2"])
df["Label"] = rf.predict(X_test)
df_l0 = df[df.Label == 0]
df_l1 = df[df.Label == 1]
axs[0].plot(decision[0], decision[1], "b*", label="Decision")
axs[0].scatter(df_l0["F1"].values, df_l0["F2"].values, alpha=0.4)
axs[0].scatter(df_l1["F1"].values, df_l1["F2"].values, alpha=0.4)
axs[0].set_title("(a) Model Prediction")
axs[0].set_xlim(min(df["F1"].values) + 0.010, max(df["F1"].values) - 0.010)
axs[0].set_ylim(min(df["F2"].values) + 0.010, max(df["F2"].values) - 0.010)
axs[0].set_xlabel("F1")
axs[0].set_ylabel("F2")

# Initializes explainer
exp = Explanation(X_train, ["F1", "F2"], None,
                  y_train, "Label", None,
                  model=RandomForestClassifier(n_estimators=100),
                  explainer=LogisticRegressionExplainer,
                  sampler=GaussianExponentialSampler,
                  depicter=DepicterBarWeights)

linspace = np.linspace(0.005, 0.5, 100)
metric = []
weight = [[], []]

for i in linspace:
    tmp = exp.sample_explain_depict(decision, num_samples=5000, measure=i, depict=False)
    metric.append(tmp['metric'])

    weight[0].append(abs(tmp['weights'][0][1])/(abs(tmp['weights'][0][1]) + abs(tmp['weights'][1][1])))
    weight[1].append(abs(tmp['weights'][1][1])/(abs(tmp['weights'][0][1]) + abs(tmp['weights'][1][1])))

    # # Plots regression line
    # xs = np.array([min(df["F1"].values), max(df["F1"].values)])
    # ys = (-tmp['weights'][0][1] * xs - tmp['weights'][2][1]) / tmp['weights'][1][1]
    # axs[0].plot(xs, ys)

axs[1].set_title("(b) Weighted Accuracy")
axs[1].plot(linspace, metric,  '--', label="Acc")
axs[1].set_xlabel("$\ell$")

axs[2].set_title("(c) Feature Values (\%)")
axs[2].plot(linspace, weight[0], label="F1")
axs[2].plot(linspace, weight[1], label="F2")
axs[2].set_xlabel("$\ell$")

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.suptitle("Multiples Explanations for Values of $\ell$")
plt.subplots_adjust(wspace=0.25, top=0.7)
plt.savefig("./imgs/test_logistic_regression_robust_explanation_1.pdf", bbox_inches="tight")

