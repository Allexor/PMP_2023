from scipy import stats
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # pct 1
    N = 200  # 200 de timpi medii sau 200 de clienti si calculam pt fiecare timpul mediu
    mu = 2  # am luat media de 2min
    sigma = 0.5  # deviatia de 0.5min
    timp = stats.norm.rvs(mu, sigma, size=N)

    # pct 2 si 3
    with pm.Model() as model:
        mu_apr = 2  # apriori mu
        sigma_apr = 0.5  # apriori sigma
        aposteriori_mu = pm.Normal('apost_mu', mu=mu_apr, sigma=sigma_apr)  # distrib normala cu mu si sigma

    with model:
        data = pm.sample(1000, return_inferencedata=True)
        az.plot_posterior(data, var_names='apost_mu')
    plt.show()
    # este in jur de 2 cum ne-am asteptat
