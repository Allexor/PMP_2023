import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az
import pymc as pm

#ex 1
def posterior_grid(grid_points=50, heads=6, tails=9):
    grid = np.linspace(0, 1, grid_points)
    # prior = np.repeat(1 / grid_points, grid_points)  # uniform prior
    # prior = (grid <= 0.5).astype(int)
    prior = abs(grid - 0.5)  # si alte valori
    likelihood = stats.binom.pmf(heads, heads + tails, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior


def calc_pi(N):
    x, y = np.random.uniform(-1, 1, size=(2, N))
    inside = (x ** 2 + y ** 2) <= 1
    pi = inside.sum() * 4 / N
    error = abs((pi - np.pi) / pi) * 100
    return error


def metropolis(func, draws=10000):
    trace = np.zeros(draws)
    old_x = 0.5  # func.mean()
    old_prob = func.pdf(old_x)
    delta = np.random.normal(0, 0.5, draws)
    for i in range(draws):
        new_x = old_x + delta[i]
        new_prob = func.pdf(new_x)
        acceptance = new_prob / old_prob
        if acceptance >= np.random.random():
            trace[i] = new_x
            old_x = new_x
            old_prob = new_prob
        else:
            trace[i] = old_x
    return trace


if __name__ == '__main__':
    data = np.repeat([0, 1], (100, 30))  # si alte valori
    points = 10
    h = data.sum()
    t = len(data) - h
    grid, posterior = posterior_grid(points, h, t)
    plt.plot(grid, posterior, 'o-')
    plt.title(f'heads = {h}, tails = {t}')
    plt.yticks([])
    plt.xlabel('θ')
    plt.show()

    # ex2
    """
    N = 10000
    x, y = np.random.uniform(-1, 1, size=(2, N))
    inside = (x ** 2 + y ** 2) <= 1
    pi = inside.sum() * 4 / N
    error = abs((pi - np.pi) / pi) * 100
    outside = np.invert(inside)
    plt.figure(figsize=(8, 8))
    plt.plot(x[inside], y[inside], 'b.')
    plt.plot(x[outside], y[outside], 'r.')
    plt.plot(0, 0, label=f'π*= {pi:4.3f} error = {error: 4.3f}', alpha=0)
    plt.axis('square')
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc=1, frameon=True, framealpha=0.9)
    plt.show()
    """
    N = [100, 1000, 10000]
    err100 = []
    err1000 = []
    err10000 = []
    for i in range(0, 1000):
        err_res = calc_pi(100)
        err100.append(err_res)
    for i in range(0, 1000):
        err_res = calc_pi(1000)
        err1000.append(err_res)
    for i in range(0, 1000):
        err_res = calc_pi(10000)
        err10000.append(err_res)
    print("err100: ", err100, "\nerr1000: ", err1000, "\nerr10000: ", err10000)

    media100 = np.mean(err100)
    stdev100 = np.std(err100)
    media1000 = np.mean(err1000)
    stdev1000 = np.std(err1000)
    media10000 = np.mean(err10000)
    stdev10000 = np.std(err10000)
    medii = [media100, media1000, media10000]
    stdev = [stdev100, stdev1000, stdev10000]
    plt.figure(figsize=(8, 8))
    plt.errorbar(N, medii, yerr=stdev)
    plt.xlabel('N')
    plt.ylabel('err')
    plt.show()

    # ex 3 incomplet (model betabinomial c3 slide 18/32)

    func = stats.beta(2, 5)
    trace = metropolis(func=func)
    x = np.linspace(0.01, .99, 100)
    y = func.pdf(x)
    plt.xlim(0, 1)
    plt.plot(x, y, 'C1-', lw=3, label='True distribution')
    plt.hist(trace[trace > 0], bins=25, density=True, label='Estimated distribution')
    plt.xlabel('x')
    plt.ylabel('pdf(x)')
    plt.yticks([])
    plt.legend()
    plt.show()
