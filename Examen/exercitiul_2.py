import pymc as pm
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az

if __name__ == '__main__':
    x = np.random.geometric(p=0.3, size=10000)
    y = np.random.geometric(p=0.5, size=10000)
    conditie = x > y ** 2