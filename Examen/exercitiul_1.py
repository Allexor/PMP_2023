import pymc as pm
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az

# nu am pus screenshoturi deoarece nu am avut timp cu executia pt ca a durat foarte mult si nu s-a terminat

# NITA ALEXANDRU B!

if __name__ == '__main__':

    # a

    data = pd.read_csv('Titanic.csv')
    pclass = data['Pclass'].values
    age = data['Age'].values.astype(float)
    for i in range(0, len(age)):
        if age[i] > 0 or age[i] < 99:
            continue
        else:
            age[i] = 0  # unde erau NaN, varsta lipsa am pus valoarea 0
    survived = data['Survived'].values
    # print(pclass)
    # print(age)
    # print(survived)

    # b

    # modelul  = alpha + beta1 * x1(pclass) + beta2 *x2(age)
    with pm.Model() as model:
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta1 = pm.Normal('beta1', mu=0, sigma=1)
        beta2 = pm.Normal('beta2', mu=0, sigma=1)
        eps = pm.HalfCauchy('eps', 5)
        miu = pm.Deterministic('miu', alpha + beta1 * pclass + beta2 * age)
        surv_pred = pm.Normal('surv_pred', mu=miu, sigma=eps, observed=survived)  # vedem survived
        idata = pm.sample(2000, tune=2000, return_inferencedata=True)

    az.plot_trace(idata, var_names=['alpha', 'beta1', 'beta2', 'eps'])
    plt.show()

    # c

    az.plot_forest(idata, var_names=['beta1', 'beta2'])
    plt.show()
    summary = az.summary(idata, var_names=['beta1', 'beta2'])
    # iar apoi din grafic si output am putea observa care variabila are o influenta mai mare, dar nu am graficul
    # pentru ca dureaza mult rularea... . in teorie ne-am uita pe grafic ce variabila este mai mare => influenteaza
    # mai mult (daca nu sunt alte exceptii)

    # d - o incercare

    posterior_g = idata.posterior.stack(samples={"chain", "draw"})  # pe PC nu am versiunea de pymc care suporta
    mu = posterior_g['alpha'] + 2 * posterior_g[beta1] + 30 * posterior_g[beta2]  # acest format cu .posterior
    az.plot_posterior(mu.values, hdi_prob=0.9)

    # sau

    pm.set_data({"pclass": 2, "age": 30}, model=model)
    ppc = pm.sample_posterior_predictive(idata, model=model)
    posterior_predictive = ppc['posterior_predictive']
    az.plot_hdi(posterior_predictive['surv_pred'], hdi_prob=0.90)
    plt.show()
