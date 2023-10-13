import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import arviz as az

server = np.random.choice([1,2,3,4], size=10000, p=[0.25, 0.25, 0.30, 0.20]) #alegem un server cu probabilitatea coresp

server1 = np.where(server == 1, stats.gamma.rvs(4, scale=1/3, size=10000) , 0)  #daca a fost ales acest server, calculam, daca nu va lua val0
server2 = np.where(server == 2, stats.gamma.rvs(4, scale=1/2, size=10000) , 0) 
server3 = np.where(server == 3, stats.gamma.rvs(5, scale=1/2, size=10000) , 0) 
server4 = np.where(server == 4, stats.gamma.rvs(5, scale=1/3, size=10000) , 0) 

x = server1 + server2 + server3 + server4 + stats.expon.rvs(scale = 1/4, size=10000) #adaugam si latenta

nr=0
for i in x:
    if i > 3:
        nr = nr+1
print("Probabilitatea timp > 3: ", nr/10000)

az.plot_posterior({"x":x})
plt.show()
