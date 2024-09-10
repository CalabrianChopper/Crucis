############################################
# Importo librerie
############################################

import random
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import matplotlib.patches as mpatches
import numpy as np

############################################
# Condizione di arresto
############################################

class StopCondition(StopIteration):
    pass

############################################
# Simulazione
############################################

class Simulation:
    def __init__(self, G, initial_state, state_transition, alpha, beta, pvacc, initial_vacc_rate, stop_condition=None, name='', pos=None):
        self.G = G
        self._initial_state = initial_state
        self._state_transition_function = state_transition
        self._stop_condition = stop_condition
        self._pos = pos or nx.random_layout(G)
        self.alpha = alpha
        self.beta = beta
        self.pvacc = pvacc
        self.initial_vacc_rate = initial_vacc_rate
        if stop_condition and not callable(stop_condition):
            raise TypeError("'stop_condition' should be a function")
        self.name = name or 'Simulation'

        self._states = []
        self._value_index = {}
        self._cmap = plt.colormaps['tab10']

        self._initialize()
        
        
    def _append_state(self, state):
        self._states.append(state)
        for value in set(node_state['status'] for node_state in state.values()):
            if value not in self._value_index:
                self._value_index[value] = len(self._value_index)
                
                
    def _initialize(self):
        if self._initial_state:
            if callable(self._initial_state):
                state = self._initial_state(self.G, self.initial_vacc_rate)
            else:
                state = self._initial_state
            for n in self.G.nodes():
                nx.set_node_attributes(self.G, state, 'state')

        if any(self.G.nodes[n].get('state') is None for n in self.G.nodes):
            raise ValueError('All nodes must have an initial state')

        self._append_state(state)
        
        
    def _step(self):
        state = nx.get_node_attributes(self.G, 'state')
        if self._stop_condition and self._stop_condition(self.G, state):
            raise StopCondition
        
        new_state = {}
        new_state = self._state_transition_function(self.G, state, self.alpha, self.beta, self.pvacc)
        
        state.update(new_state)
        nx.set_node_attributes(self.G, state, 'state')
        self._append_state(state)
        
        
    def _categorical_color(self, value):
        color_map = {
            'S': 'yellow',
            'I': 'red',
            'R': 'lightgreen',
            'V': 'darkgreen'
        }
        return color_map.get(value['status'], 'gray')  # default to gray
    
    
    @property
    def steps(self):
        return len(self._states) - 1

    def state(self, step=-1):
        try:
            return self._states[step]
        except IndexError:
            raise IndexError('Simulation step %i out of range' % step)

    def draw(self, step=-1, labels=None, **kwargs):
        state = self.state(step)
        node_colors = [self._categorical_color(state[n]) for n in self.G.nodes]
        nx.draw(self.G, pos=self._pos, node_color=node_colors, **kwargs)

        if labels is None:
            labels = sorted(set(state.values()), key=self._value_index.get)
        patches = [mpatches.Patch(color=self._categorical_color(l), label=l) for l in labels]
        plt.legend(handles=patches)

        if step == -1:
            step = self.steps
        if step == 0:
            title = 'initial state'
        else:
            title = 'step %i' % (step)
        if self.name:
            title = '{}: {}'.format(self.name, title)
        plt.title(title)
        
    def plot(self, min_step=None, max_step=None, labels=None, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        x_range = range(min_step or 0, max_step or len(self._states))
        counts = [Counter(s['status'] for s in state.values()) for state in self._states[min_step:max_step]]
        if labels is None:
            labels = {k for count in counts for k in count}
            labels = sorted(labels, key=self._value_index.get)

        color_map = {
            'S': 'yellow',
            'I': 'red',
            'R': 'lightgreen',
            'V': 'darkgreen'
        }

        for label in labels:
            series = [count.get(label, 0) / sum(count.values()) for count in counts]
            ax.plot(x_range, series, label=label, color=color_map.get(label, 'gray'), **kwargs)

        title = 'node state proportions'
        if self.name:
            title = '{}: {}'.format(self.name, title)
        ax.set_title(title)
        ax.set_xlabel('Simulation step')
        ax.set_ylabel('Proportion of nodes')
        ax.legend()
        ax.set_xlim(x_range.start)
        
        return ax    
    def run(self, steps=1):
        for _ in range(steps):
            try:
                self._step()
            except StopCondition as e:
                print("Stop condition met at step %i." % self.steps)
                break
        return self
    
############################################
# Stato iniiziale dei nodi e pazient zero
############################################
def initial_state(G, initial_vacc_rate):
    state = {}
    for node in G.nodes:
        if random.random() < initial_vacc_rate:
            state[node] = {'status': 'V', 'age': random.randint(0, 100)}
        else:
            state[node] = {'status': 'S', 'age': random.randint(0, 100)}
    
    eligible_nodes = [node for node in G.nodes if state[node]['age'] <= 20 and state[node]['status'] == 'S']
    if eligible_nodes:
        patient_zero = random.choice(eligible_nodes)
        state[patient_zero]['status'] = 'I'
    
    return state
############################################
# Transizioni di stato modello SIRV
############################################
    
def state_transition_SIRV(G, current_state, alpha, beta, pvacc):
    ALPHA = float(alpha)
    BETA = float(beta)
    PVACC = float(pvacc)

    next_state = {}
    
    for node in G.nodes:
        current_status = current_state[node]['status']
        age = current_state[node]['age']
        
        if current_status == 'I':
            if random.random() < BETA:
                next_state[node] = {'status': 'R', 'age': age}
        else:
            if current_status == 'S':
                if random.random() < PVACC:
                    next_state[node] = {'status': 'V', 'age': age}
                elif age <= 20: 
                    for neighbor in G.neighbors(node):
                        if current_state[neighbor]['status'] == 'I':
                            if random.random() < ALPHA:
                                next_state[node] = {'status': 'I', 'age': age}
                                break
    return next_state

############################################
# Ricostruzione grafo da file csv
############################################

def ricostruisci_grafo(nodi_file, archi_file):
    df_nodi = pd.read_csv(nodi_file)
    df_archi = pd.read_csv(archi_file)

    G = nx.Graph()

    for nodo in df_nodi['nodo']:
        G.add_node(nodo)

    for _, row in df_archi.iterrows():
        G.add_edge(row['nodo_origine'], row['nodo_destinazione'])
    return G

G = ricostruisci_grafo('nodi.csv', 'archi.csv')

############################################
# Definizione parametri modello SIRV
############################################

alpha = 0.9
beta = 0.2
pvacc = 0.01
initial_vacc_rate = 0.5

############################################
# Creazione grafo SBM customizzato
############################################

# def create_large_sbm_graph(n_nodes=2000, n_groups=10, p_intra=0.1, p_inter=0.001):
#     group_sizes = [n_nodes // n_groups] * n_groups
#     remainder = n_nodes % n_groups
#     for i in range(remainder):
#         group_sizes[i] += 1

#     prob_matrix = np.full((n_groups, n_groups), p_inter)
#     np.fill_diagonal(prob_matrix, p_intra)

#     G = nx.stochastic_block_model(group_sizes, prob_matrix, seed=42)

#     return G

# G = create_large_sbm_graph()

# sizes = [75, 75, 300]
# probs = [[0.25, 0.05, 0.02], [0.05, 0.35, 0.07], [0.02, 0.07, 0.40]]
# G = nx.stochastic_block_model(sizes, probs, seed=0)

#G = nx.complete_graph(2000)
# G.draw()

############################################
# Avvio simulazione
############################################


sim = Simulation(G, initial_state, state_transition_SIRV, alpha, beta, pvacc, initial_vacc_rate=initial_vacc_rate)
simulation_result = sim.run(10)
############################################
# Creazione file csv per grafo
############################################

lista_nodi = list(G.nodes)
df_nodi = pd.DataFrame({'nodo': lista_nodi})
df_nodi.to_csv('nodi.csv', index=False)

lista_archi = list(G.edges)
df_archi = pd.DataFrame(lista_archi, columns=['nodo_origine', 'nodo_destinazione'])
df_archi.to_csv('archi.csv', index=False)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10)) 

############################################
# Ploot simulazione e grafico finale
############################################

simulation_result.plot(ax=ax1)
ax1.set_title("Simulation Progress")
final_state = simulation_result.state()
color_map = {'S': 'yellow', 'I': 'red', 'R': 'lightgreen', 'V': 'darkgreen'}
node_colors = [color_map.get(final_state[node]['status'], 'gray') for node in G.nodes()]

nx.draw(G, ax=ax2, with_labels=True, node_color=node_colors, node_size=300, font_size=8, font_weight='bold')
ax2.set_title("Final Graph Structure")
ax2.axis('off')

legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=state,
                    markerfacecolor=color, markersize=10)
                    for state, color in color_map.items()]
ax2.legend(handles=legend_elements, loc='best')

# Calculate percentages
total_nodes = len(G.nodes())
state_counts = Counter(final_state[node]['status'] for node in G.nodes())
percentages = {state: count / total_nodes * 100 for state, count in state_counts.items()}

# Create new legend elements with percentages
percentage_legend_elements = [f"{state}: {percentages[state]:.1f}%" for state in color_map if state in percentages]

# Add new legend below the graph
ax2.text(0.5, -0.1, "\n".join(percentage_legend_elements), 
         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),
         transform=ax2.transAxes, ha='center', va='center')

plt.tight_layout()
plt.show()

