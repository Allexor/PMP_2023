import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import arviz as az

mecanic = np.random.choice([1,2], size = 10000, p=[0.4, 0.6]) # alegem mecanicul

x=np.where(mecanic == 1, stats.expon.rvs(scale= 1/4, size=10000), stats.expon.rvs(scale= 1/6, size=10000)) #in functie de mecanicul ales vedem timp servire exponential

media_x = np.mean(x)
stdv_x = np.std(x)

print("Media: ", media_x)
print("Deviatia: ", stdv_x)

az.plot_posterior({'x':x})
#plt.hist(x)
plt.show()