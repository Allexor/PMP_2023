import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# am sters manual liniile cu '?' din fisierul cu date
if __name__ == '__main__':
    ans = pd.read_csv('auto-mpg.csv')
    # print(ans)

    plt.scatter(ans['horsepower'], ans['mpg'])
    plt.xlabel('cp')
    plt.ylabel('mpg')
    plt.show()

    with pm.Model() as model:
        cp = ans['horsepower'].values
        mpg = ans['mpg'].values

        alpha = pm.Normal('alpha', mu=0, sigma=1)
        beta = pm.Normal('beta', mu=0, sigma=1)
        mu = alpha + beta * cp

        pred = pm.Normal('pred', mu=mu, sigma=1, observed=mpg)
        idata_g = pm.sample(2000, return_inferencedata=True)

    # e aproximativ ca in curs(slide18) doar ca au trebuit modificari din cauza diferitelor
    # versiuni de pymc2 si pymc3 si de acolo apar erori
    # dureaza cateva minute bune sample-ul
    alpha_p = idata_g['alpha']
    beta_p = idata_g['beta']
    alpha_m = alpha_p.mean()
    beta_m = beta_p.mean()
    plt.scatter(cp, mpg, alpha=0.5)
    for i, j in zip(alpha_p, beta_p):
        plt.plot(cp, i + j * cp, c='gray', alpha=0.5)
    plt.plot(cp, alpha_m + beta_m * cp, color='k', label=f'y = {alpha_m:.2f} + {beta_m:.2f} * cp')
    plt.xlabel('cp')
    plt.ylabel('mpg')
    plt.legend()
    plt.show()

    # print("cp", cp)
    # print("mpg", mpg)
