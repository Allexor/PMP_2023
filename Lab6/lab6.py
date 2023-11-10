import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

if __name__ == '__main__':
    Y = [0, 5, 10]
    THETA = [0.2, 0.5]

    for i in Y:
        for j in THETA:
            with pm.Model() as model:
                n = pm.Poisson('n', mu=10)
                aposteriori = pm.Binomial('apost', n=n, p=j, observed=i)

            with model:
                data = pm.sample(1000, return_inferencedata=True)
                az.plot_posterior(data, var_names='n')
    plt.show()
