from scipy import stats

#ex 1
def ex1(alpha):
    lambdaa = 20
    N = stats.poisson.rvs(lambdaa)
    print("nr clienti: ", N)

    mu = 2
    sigma = 0.5
    T = stats.norm.rvs(mu, sigma, size = N)
    print("timp clienti plasare si plata: ", T)

    G = stats.expon.rvs(scale = alpha, size = N)
    print("timp clienti gatire comanda", G)

    #return adaug pentru ex2 si ex3

ex1(11) #alpha = 11 o valoare random momentan pt o rulare de verif. ex 1
