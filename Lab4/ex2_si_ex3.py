from scipy import stats


#ex 2 - P(Ti + Gi <= 15, oricare i= 1,N) >= 95%
def toti_clienti_serviti_timp(alpha):
    lambdaa = 20
    N = stats.poisson.rvs(lambdaa)
    #print("nr clienti: ", N)

    mu = 2
    sigma = 0.5
    T = stats.norm.rvs(mu, sigma, size = N)
    #print("timp clienti plasare si plata: ", T)

    G = stats.expon.rvs(scale = alpha, size = N)
    #print("timp clienti gatire comanda", G)
    
    nr_clienti_serviti_la_timp = 0
    for i in range(0, N):
        if T[i] + G[i] <= 15:
            nr_clienti_serviti_la_timp = nr_clienti_serviti_la_timp + 1

    if nr_clienti_serviti_la_timp == N: #toti clientii sunt serviti sub 15
        #print(nr_clienti_serviti_la_timp, N)
        return True
    else:
        return False

alpha_mediu = 0 
for i in range(0, 15):
    # o simulare. noi vrem un alpha mediu asa ca rulam de 15 ori ca sa aflam o medie (se poate si mai mult dar dureaza)
    alpha = 20
    while alpha >= 0:
        cnt = 0
        for j in range (0,50):
            if toti_clienti_serviti_timp(alpha) is True:
                cnt = cnt + 1
        if cnt / 50 >= 0.95:
            print("simulare: ", i+1, "am gasit alpha  = ", alpha, " cu Probabilitate = ", cnt / 50)
            break
        else:
            alpha = alpha - 0.1
    alpha_mediu =  alpha_mediu + alpha
alpha_mediu = alpha_mediu / 15   
print("\n>>>Un alpha mediu a.i toti clientii sa fie serviti sub 15min: ", alpha_mediu)  
        

#exercitiu 3

def timp_mediu_asteptare(alpha):
    lambdaa = 20
    N = stats.poisson.rvs(lambdaa)
    mu = 2
    sigma = 0.5
    T = stats.norm.rvs(mu, sigma, size = N)
    G = stats.expon.rvs(scale = alpha, size = N)

    timp_total = 0
    for i in range(0, N):
        timp_total = timp_total + T[i] + G[i]
    
    return timp_total/N

timp_mediu = 0 
for i in range(0, 35): #facem si aici o medie
    timp_mediu = timp_mediu + timp_mediu_asteptare(alpha_mediu)
timp_mediu = timp_mediu / 35
print(">>>Timp mediu de asteptare client cu alpha aflat la ex2: ", timp_mediu)