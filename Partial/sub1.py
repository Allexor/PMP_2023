import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# subpct1  este rezolvat corect, la subpct2(implicit si subpct3) am o eroare
if __name__ == '__main__':
    # PCT 1 - rezolvat
    castigatori = []
    for i in range(0, 20000):  # simulam de 20000 de ori jocul
        n = 0
        m = 0
        jucator = np.random.choice([0, 1], p=[0.5, 0.5])  # alegem cine incepe jocul P0 sau P1

        if jucator == 0:
            # runda1
            moneda_juc0 = np.random.choice([1, 2], p=[1 / 3, 2 / 3])  # 1=stema, 2=ban, moneda masluita
            if moneda_juc0 == 1:  # daca a picat stema
                n = n + 1  # adunam 1
            # runda2
            # apoi celalalt jucator arunca de n+1
            for j in range(0, n + 1):
                moneda_juc1 = np.random.choice([1, 2], p=[0.5, 0.5])  # 1=stema, 2=ban, moneda normala
                if moneda_juc1 == 1:
                    m = m + 1

        if jucator == 1:  # acelasi lucruri ca mai sus doar ca jucatorul care incepe este p1
            # runda1
            moneda_juc1 = np.random.choice([1, 2], p=[0.5, 0.5])
            if moneda_juc1 == 1:
                n = n + 1
            # runda2
            for j in range(0, n + 1):
                moneda_juc0 = np.random.choice([1, 2], p=[1 / 3, 2 / 3])
                if moneda_juc0 == 1:
                    m = m + 1

        if n >= m:
            castigatori.append(0)
        else:
            castigatori.append(1)  # am pus intr-o variabila toate rezultatele din cele 20000 de simulari
    p0_castig = 0
    p1_castig = 0
    for i in castigatori:  # vedem care jucator a castigat de cele mai multe ori
        if i == 0:
            p0_castig = p0_castig + 1
        if i == 1:
            p1_castig = p1_castig + 1
    if p0_castig > p1_castig:
        print(">>PUNCT 1 : Sansa de castig mai mare o are p0. Nr castiguri p0", p0_castig, "vs p1", p1_castig)
    elif p0_castig < p1_castig:
        print(">>PUNCT 1: Sansa de castig mai mare o are p1. Nr castiguri p1", p0_castig, "vs", p1_castig)
    else:
        print(">>PUNCT 1: Au aceeasi sansa de castig. Nr castiguri p1", p0_castig, "vs", p1_castig)

    # PCT2 - o incercare.., cred ca nu am gandit bine reteaua bayesiana
    model = BayesianNetwork([('R1P0', 'R2P1'), (
        'R1P1', 'R2P0')])  # avem o varianta in care runda1 arunca p0 si una in care in runda1 arunca p1
    cpd_R1P0 = TabularCPD(variable='R1P0', variable_card=2, values=[[1 / 3], [2 / 3]])  # 1/3 stema, 2/3 ban
    cpd_R2P1 = TabularCPD(variable='R2P1', variable_card=2, values=[[2 * 0.5, 0.5],
                                                                    [2 * 0.5, 0.5]], evidence=['R1P0'])
    # aici la r2p1 trebuie sa luam daca in runda1 P0 a obtinut stema sau nu
    # am gandit momentan sub forma ca daca a picat stema in prima, el va arunca de 2 ori asa ca am pus de forma 2*../2
    # daca nu, arunca doar o data si e prob clasica
    # la fel si in urmatorul caz
    cpd_R1P1 = TabularCPD(variable='R1P1', variable_card=2, values=[[0.5], [0.5]])  # 0.5 stema, 0.5 ban
    cpd_R2P0 = TabularCPD(variable='R2P0', variable_card=2, values=[[2 * 2 / 3, 1 / 3],
                                                                    [2 * 1 / 3, 1 / 3]], evidence=['R1P1'])
    model.add_cpds(cpd_R1P0, cpd_R2P1, cpd_R1P1, cpd_R2P0)
    assert model.check_model()

    # PCT3 - o incercare..
    infer = VariableElimination(model)
    result = infer.query(variables=['R1P1'], evidence={'R2P0': 0})  # cazul in care runda1 arunca p1
    print(result)

    infer2 = VariableElimination(model)
    result2 = infer.query(variables=['R1P0'], evidence={'R2P1': 0})
    print(result2)
