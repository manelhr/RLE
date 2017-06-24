import numpy as np
import pandas as pd
from deap import algorithms, base, benchmarks, tools, creator


def nsga2_robust_explanation():

    toolbox = base.Toolbox()

    # creates fitness function
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))

    # register type of individual
    creator.create("Individual", list, fitness=creator.FitnessMulti)
