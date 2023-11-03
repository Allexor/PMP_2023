import numpy as np
import pymc as pm


count_data = np.genfromtxt("trafic.csv", delimiter = ",", dtype = int, skip_header = 1)
print("Datele: \n", count_data)

count_data_valori = []
for i in count_data:
    count_data_valori.append(i[1])
#print("Doar valori: ", count_data_valori)


with pm.Model() as model:
    alpha = 1.0 / count_data.mean()

    lambda_1 = pm.Exponential("lambda_1", alpha)
    lambda_2 = pm.Exponential("lambda_2", alpha)
    lambda_3 = pm.Exponential("lambda_3", alpha)
    lambda_4 = pm.Exponential("lambda_4", alpha)
    lambda_5 = pm.Exponential("lambda_5", alpha)

    #min0 start ora 7: min180, ora 8: min240, ora 16: min720, ora19: min900, , min 1200 final 

    tau1 = pm.DiscreteUniform("tau1", lower = 160, upper = 200) #in jur de 180
    tau2 = pm.DiscreteUniform("tau2", lower = 220, upper = 260) #in jur de 240
    tau3 = pm.DiscreteUniform("tau3", lower = 700, upper = 740) #in jur de 720
    tau4 = pm.DiscreteUniform("tau4", lower = 880, upper = 920) #in jur de 900

    lambda_ = pm.math.switch(tau1 > tau2, lambda_1, 
                             pm.math.switch(tau2 > tau3, lambda_2, 
                                            pm.math.switch(tau3 > tau4, lambda_3, 
                                                           pm.math.switch(tau4 > len(count_data_valori), lambda_4, lambda_5))))
with model:
    observation = pm.Poisson("obs", lambda_, observed = count_data_valori)

with model:
    step = pm.Metropolis()
    trace = pm.sample(10, tune = 5, step = step, return_inferencedata = False, cores = 1) 
    #valori mici ca sa mearga repede pe un core, pe mai multe cores primesc eroare
    #se poate incerca: sters cores = 1 sa mearga pe mai multe cores si pus valori mai normale(de ex 1000, 500)


mapp = pm.find_MAP(model = model)
tau1_samples = mapp['tau1']
tau2_samples = mapp['tau2']
tau3_samples = mapp['tau3']
tau4_samples = mapp['tau4']
lambda_1_samples = mapp['lambda_1']
lambda_2_samples = mapp['lambda_2']
lambda_3_samples = mapp['lambda_3']
lambda_4_samples = mapp['lambda_4']
lambda_5_samples = mapp['lambda_5']

intervale = [tau1_samples, tau2_samples, tau3_samples, tau4_samples, len(count_data_valori)]
valori_lambda = [lambda_1_samples, lambda_2_samples, lambda_3_samples, lambda_4_samples, lambda_5_samples]

print("intervale: ", intervale)
print("valori_lambda: ", valori_lambda)