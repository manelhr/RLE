from rle.explainers.logistic_regression_explainer import LogisticRegressionExplainer
from rle.samplers.gaussian_exponential_sampler import GaussianExponentialSampler
from rle.depicters.depicter_bar_weights import DepicterBarWeights
from sklearn.model_selection import train_test_split
from rle.explanation.robust_explanation import RobustExplanation
from sklearn.ensemble import RandomForestClassifier
import sklearn.datasets as datasets
import matplotlib.pyplot as plt
import seaborn as sns
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


# Initializes double triangle dataset
X = np.random.rand(5000, 2)
triangle = np.array([[0.4, 0.4], [0.6, 0.6], [0.01, 0.99]])
decision = np.array([0.27, 0.64])
y = np.array(list(map(lambda x: int(x[1] < x[0] or inside_triangle(x, triangle[0], triangle[1], triangle[2])), X)))

# Plots classified data (Random Forest)
X_train, X_test, y_train, y_test = train_test_split(X, y)
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

# Initializes explainer
exp = RobustExplanation(X_train, ["F1", "F2"], None,
                        y_train, "Label", None,
                        model=RandomForestClassifier(n_estimators=100),
                        explainer=LogisticRegressionExplainer,
                        sampler=GaussianExponentialSampler,
                        depicter=DepicterBarWeights)

exp.sample_explain_depict(decision, 5, num_samples=5000, measure_min=0.005, measure_max=0.5, number_eval=500)

