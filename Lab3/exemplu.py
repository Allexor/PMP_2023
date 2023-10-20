from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx

# Defining the model structure. We can define the network by just passing a list of edges.
model = BayesianNetwork([('R', 'C'), ('U', 'C')])

# Defining individual CPDs.
cpd_r = TabularCPD(variable='R', variable_card=2, values=[[0.8], [0.2]]) # R=0 full price, R=1 discount
cpd_u = TabularCPD(variable='U', variable_card=2, values=[[0.95], [0.05]]) # U=0 no urgent need, U=1 urgent need

# The CPD for C is defined using the conditional probabilities based on U and R
cpd_c = TabularCPD(variable='C', variable_card=2, 
                   values=[[0.8, 0.5, 0.2, 0.0], 
                           [0.2, 0.5, 0.8, 1.0]],
                  evidence=['U', 'R'],
                  evidence_card=[2, 2])

# Associating the CPDs with the network
model.add_cpds(cpd_r, cpd_u, cpd_c)

# Verifying the model
assert model.check_model()

# Performing exact inference using Variable Elimination
infer = VariableElimination(model)
result = infer.query(variables=['U'], evidence={'C': 1})
print(result)

pos = nx.circular_layout(model)
nx.draw(model, pos=pos, with_labels=True, node_size=4000, font_weight='bold', node_color='skyblue')
plt.show()