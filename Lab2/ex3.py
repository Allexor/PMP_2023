import numpy as np
import matplotlib.pyplot as plt
import arviz as az

ss, sb, bs, bb = [], [], [], [] #vectori pentru cele 100 de rezultate)

for i in range(100): # "100 rezultate independente"
    ss_per_experiment, sb_per_experiment, bs_per_experiment, bb_per_experiment = 0, 0, 0, 0 
    # contorizam pentru fiecare experiment(din cele 100) de cate ori pica stema/ban pt fiecare moneda, aruncate de 10 ori/experiment

    moneda1 = np.random.choice([1,2], size=10, p=[0.5,0.5])#nemasluita 1-stema, 2-ban, size=10 aruncari
    moneda2 = np.random.choice([1,2], size=10, p=[0.3,0.7])#masluita,  1-stema, 2-ban, size=10 aruncari

    print("debug -- moneda1: ",  moneda1)
    print("debug -- moneda2: ",  moneda2)
    print("= = = = = = = = = = = = = = = = = = = = =")

    for j in range(10): # parcurgem cele 10 aruncari
        if moneda1[j]==1 and moneda2[j]==1: #stema stema
            ss_per_experiment = ss_per_experiment + 1
        elif moneda1[j]==1 and moneda2[j]==2: #stema ban
            sb_per_experiment = sb_per_experiment + 1
        elif moneda1[j]==2 and moneda2[j]==1: #ban stema
            bs_per_experiment = bs_per_experiment + 1
        else:
            bb_per_experiment = bb_per_experiment + 1

    ss.append(ss_per_experiment) #adaugam in vector rezultatele
    sb.append(sb_per_experiment)
    bs.append(bs_per_experiment)
    bb.append(bb_per_experiment)

az.plot_posterior({'ss':ss, 'sb':sb, 'bs':bs, 'bb':bb})
plt.show()