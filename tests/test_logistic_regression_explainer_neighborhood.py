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

sns.set_style("whitegrid")
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

fig, axs = plt.subplots(1, 5, figsize=(10, 2), sharex=True, sharey=True)

# Initializes dataset and variables
decision = np.array([0.34, 0.44])
measures = [0.085, 0.10, 0.15, 0.50, 2]
sample_size = 250

for measure, i in zip(measures, range(len(axs))):
    np.random.seed(1)
    ax = axs[i]

    # Initializes peak dataset
    X = np.random.rand(800, 2)
    X[:, 0] = (X[:, 0] * 6) - 3
    X[:, 1] = X[:, 1] / 2
    y = np.array(list(map(lambda x: int(x[1] < stats.norm.pdf(x[0]) and x[0] < 0), X)))
    X[:, 0] = (X[:, 0] + 3) / 6
    X[:, 1] = X[:, 1] * 2
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    df = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])
    df["Label"] = y

    # Initializes model
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)

    # Initializes sampler
    sampler = GaussianExponentialSampler(X, ["Feature 1", "Feature 2"], ["numerical", "numerical"],
                                         y, "Label", "Categorical",
                                         sample_size, measure, rf.predict_proba)

    # Initializes explainer
    explainer = LogisticRegressionExplainer(sampler)

    # Performs Logistic Regression
    weights = explainer.explain(decision)
    last_decision, df = explainer.explanation_data()

    # Calculates exponential kernel mesh
    xk = np.arange(min(df["Feature 1"].values), max(df["Feature 1"].values), .025)
    yk = np.arange(min(df["Feature 2"].values), max(df["Feature 2"].values), .025)
    X, Y = np.meshgrid(xk, yk)
    combined = np.stack([X.ravel(), Y.ravel()], axis=-1)
    distances = pairwise_distances(combined, decision.reshape(1, -1), metric='euclidean')
    exponential_distances = np.sqrt(np.exp(-(distances ** 2) / measure ** 2))
    Z = exponential_distances.reshape(len(X), len(X[0]))
    cs = ax.contourf(X, Y, Z, alpha=0.25, cmap=plt.cm.bone)

    # Plots decision
    ax.plot(decision[0], decision[1], "b*", label="Decision")

    # Plots regression line
    xs = np.array([min(df["Feature 1"].values), max(df["Feature 1"].values)])
    ys = (-weights[0][1] * xs - weights[2][1]) / weights[1][1]
    ax.plot(xs, ys)

    # Plots gaussian sample
    df_l0 = df[df.Label == 0]
    df_l1 = df[df.Label == 1]

    ax.scatter(df_l0["Feature 1"].values, df_l0["Feature 2"].values, s=df_l0["Exp. Sim."].values*10, alpha=0.7)
    ax.scatter(df_l1["Feature 1"].values, df_l1["Feature 2"].values, s=df_l1["Exp. Sim."].values*10, alpha=0.7)

    ax.set_xlim(min(df["Feature 1"].values) + 0.010, max(df["Feature 1"].values) - 0.010)
    ax.set_ylim(min(df["Feature 2"].values) + 0.010, max(df["Feature 2"].values) - 0.010)
    ax.set_title("$\ell$: " + str(measure))

plt.suptitle("Logistic Regression Explainer with Different Neighborhoods ($s = 250$)")
plt.subplots_adjust(wspace=0.25, top=0.7)
plt.savefig("./imgs/logistic_regression_explainer_neighborhood.pdf", bbox_inches="tight")
