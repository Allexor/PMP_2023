import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import numpy as np

# dureaza cam 7min un run, rezolvat momentan ex 1,2,3
if __name__ == '__main__':
    ans = pd.read_csv("Prices.csv")
    # print(ans)
    N = len(ans)
    speed_mean = np.mean(ans['Speed'].values)
    harddrive_mean = np.mean(np.log(ans['HardDrive'].values))
    speed_centered = ans['Speed'].values - speed_mean
    harddrive_centered = np.log(ans['HardDrive'].values) - harddrive_mean
    X = np.array([speed_centered, harddrive_centered]).T

    with pm.Model() as model_mlr:
        alpha_tmp = pm.Normal('alpha_tmp', mu=0, sigma=10)
        beta = pm.Normal('beta', mu=0, sigma=1, shape=2)
        eps = pm.HalfCauchy('eps', 5)
        mu = alpha_tmp + pm.math.dot(X, beta)
        alpha = pm.Deterministic('alpha', alpha_tmp - pm.math.dot(np.array([speed_mean, harddrive_mean]), beta))
        y_pred = pm.Normal('y_pred', mu=mu, sigma=eps, observed=ans['Price'].values)
        idata_mlr = pm.sample(2000, return_inferencedata=True)

    with model_mlr:
        az.plot_posterior(idata_mlr, var_names=['alpha', 'beta', 'eps'], hdi_prob=0.95)

        """ 
        plot pt ex2 :
        pt hdi beta1,2 grafic folder
        hdi beta 0(speed)95%: [-1.6, 2.2] mean 0.31
        hdi beta 1(harddrive)95%: [-1.9, 2]  mean 0.021
        """

        """
        pt ex3:
        nu cred ca sunt predictori utili deoarece ambele au mean apropiat de 0(capetele intervalelor sunt cat de cat 
        simetrice(opuse), de acolo mean asa aproape de 0). in exemplul de la curs,
        mean-ul este mult mai ridicat(0.969 si 1.469) ceea ce ar sugera ca atributele din exemplu sunt predictori utili
        contrar exercitiului nostru
        
        alta observatie este ca intervalele au capete cu acelasi semn(pozitive) => un mean mai mare,
        in cazul nostru, intervalul este cam 50% pozitiv, 50%negativ iar de aici avem un mean scazut
        """

    plt.show()
