from rle.explainers.logistic_regression_explainer import LogisticRegressionExplainer
from rle.samplers.gaussian_exponential_sampler import GaussianExponentialSampler
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

fig, axs = plt.subplots(1, 3, figsize=(10, 2.5), sharex=True, sharey=True)

# Initializes dataset and sample_size/mesh variables
decision = np.array([0.34, 0.44])
measure = 0.15
sample_sizes = np.arange(50, 500, 50)
measures = np.arange(0.05, 5, 0.1)
X_mesh, Y_mesh = np.meshgrid(sample_sizes, measures)
combined = np.stack([X_mesh.ravel(), Y_mesh.ravel()], axis=-1)
accuracy = []
feature1 = []
feature2 = []

print(len(combined))

for sample_size, measure in combined:
    np.random.seed(1)

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

    # Store results
    feature1.append(weights[0][1])
    feature2.append(weights[1][1])
    accuracy.append(explainer.metrics())

for array_v, ax, title in zip([feature1, feature2, accuracy], axs, ["Feature 1", "Feature 2", "Accuracy"]):
    Z = np.array(array_v).reshape(len(X_mesh), len(X_mesh[0]))
    cs = ax.contourf(X_mesh, Y_mesh, Z, alpha=0.7, cmap=plt.cm.bone)
    ax.set_title(title)

axs[1].set_xlabel("Sample Size ($s$)")
axs[0].set_ylabel("Neighborhood ($\ell$)")

plt.suptitle("Weights and Accuracy w.r.t. Neighborhood and Sample Size")
cbar_ax = fig.add_axes([0.85, 0.10, 0.05, 0.7])
fig.colorbar(cs, cax=cbar_ax)
plt.subplots_adjust(wspace=0.25, top=0.8, right=0.8)
plt.savefig("./imgs/logistic_regression_explainer_variation.pdf", bbox_inches="tight")
