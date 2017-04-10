import unittest
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
np.random.seed(0)

x = np.linspace(-5, 5, 1000)
y = stats.norm(0, 1).pdf(x)

plt.plot(x, y)
plt.show()
