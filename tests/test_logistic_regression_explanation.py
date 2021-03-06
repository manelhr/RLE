from rle.explainers.logistic_regression_explainer import LogisticRegressionExplainer
from rle.samplers.gaussian_exponential_sampler import GaussianExponentialSampler
from rle.depicters.depicter_bar_weights import DepicterBarWeights
from sklearn.model_selection import train_test_split
from rle.explanation.explanation import Explanation
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import seaborn as sns
import numpy as np

sns.set_style("whitegrid")
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

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

# Initializes decision
decision = np.array([0.34, 0.44])

# Initializes destination for the images
destination = "./imgs/logistic_regression_explanation.pdf"

# Initializes explainer
exp = Explanation(X_train, ["Feature 1", "Feature 2"], None,
                  y_train, "Label", None,
                  model=RandomForestClassifier(n_estimators=100),
                  explainer=LogisticRegressionExplainer,
                  sampler=GaussianExponentialSampler,
                  depicter=DepicterBarWeights,
                  destination=destination)

# Samples, explains and depict
exp.sample_explain_depict(decision, depict=True)
