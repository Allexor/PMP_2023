from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx

#dureaza cateva secunde la run dar merge(poate e doar la mine)

#ex 1
model = BayesianNetwork([('C','I'), ('C','A'), ('I','A')]) #graf cu C->I, C->A, I->A

cpd_c = TabularCPD(variable = 'C', variable_card = 2, values = [[0.9995], [0.0005]]) # 0.05% = 0.0005

cpd_i = TabularCPD(variable = 'I', variable_card = 2, values = [[0.99, 0.97], 
                                                                [0.01, 0.03]], 
                                                                evidence = ['C'], evidence_card = [2]) 
                                                                #C=0 -> 0.01, C=1-> 0.03

cpd_a = TabularCPD(variable = 'A', variable_card = 2, values = [[0.9999, 0.05, 0.98, 0.02],
                                                                [0.0001, 0.95, 0.02, 0.98]], evidence = ['C', 'I'], evidence_card=[2, 2])
                                                                #C=0I=0 ->0.0001, C=0I=1 -> 0.95, C=1I=0 -> 0.02, C=1I=1=0.98

model.add_cpds(cpd_c, cpd_i, cpd_a)
assert model.check_model()

#ex 2
infer = VariableElimination(model)
result =  infer.query(variables = ['C'], evidence = {'A': 1} )
print("Ex2:")
print(result)

#ex 3
infer = VariableElimination(model)
result =  infer.query(variables = ['I'], evidence = {'A': 0} )
print("Ex3:")
print(result)


pos = nx.circular_layout(model)
nx.draw(model, pos=pos, with_labels=True, node_size=4000, font_weight='bold', node_color='skyblue')
plt.show()

#ex 4 -bonus pdf