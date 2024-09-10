import random
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd

class StopCondition(StopIteration):
    pass

class Simulation:
    def __init__(self, G, initial_state, state_transition, alpha, beta, gamma, pvacc, stop_condition=None, name='', pos=None):
        self.G = G
        self._initial_state = initial_state
        self._state_transition_function = state_transition
        self._stop_condition = stop_condition
        self._pos = pos or nx.random_layout(G)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.pvacc = pvacc
        if stop_condition and not callable(stop_condition):
            raise TypeError("'stop_condition' should be a function")
        self.name = name or 'Simulation'

        self._states = []
        self._value_index = {}
        self._cmap = plt.cm.get_cmap('tab10')

        self._initialize()
        
    def _append_state(self, state):
        self._states.append(state)
        for value in set(state.values()):
            if value not in self._value_index:
                self._value_index[value] = len(self._value_index)

    def _initialize(self):
        if self._initial_state:
            if callable(self._initial_state):
                state = self._initial_state(self.G)
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
        new_state = self._state_transition_function(self.G, state, self.alpha, self.beta, self.gamma, self.pvacc)
        
        state.update(new_state)
        nx.set_node_attributes(self.G, state, 'state')
        self._append_state(state)
        
    def _categorical_color(self, value):
        index = self._value_index[value]
        node_color = self._cmap(index)
        return node_color

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
        patches = [mpl.patches.Patch(color=self._categorical_color(l), label=l)
                for l in labels]
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

    def plot(self, min_step=None, max_step=None, labels=None, **kwargs):
        x_range = range(min_step or 0, max_step or len(self._states))
        counts = [Counter(s.values()) for s in self._states[min_step:max_step]]
        if labels is None:
            labels = {k for count in counts for k in count}
            labels = sorted(labels, key=self._value_index.get)

        for label in labels:
            series = [count.get(label, 0) / sum(count.values()) for count in counts]
            plt.plot(x_range, series, label=label, **kwargs)

        title = 'node state proportions'
        if self.name:
            title = '{}: {}'.format(self.name, title)
        plt.title(title)
        plt.xlabel('Simulation step')
        plt.ylabel('Proportion of nodes')
        plt.legend()
        plt.xlim(x_range.start)
        
        plt.show()

        return plt.gca()

    def run(self, steps=1):
        for _ in range(steps):
            try:
                self._step()
            except StopCondition as e:
                print("Stop condition met at step %i." % self.steps)
                break
        return self
    
def initial_state(G):
    state = {node: 'S' for node in G.nodes}
    patient_zero = random.choice(list(G.nodes))
    state[patient_zero] = 'I'
    return state
    
def state_transition_SIRV(G, current_state, alpha, beta, gamma, pvacc):
    ALPHA = float(alpha)
    BETA = float(beta)
    GAMMA = float(gamma)
    PVACC = float(pvacc)

    next_state = {}
    
    for node in G.nodes:
        if current_state[node] == 'I':
            if random.random() < BETA:
                next_state[node] = 'R'
        elif current_state[node] == 'R':
            if random.random() < GAMMA:
                next_state[node] = 'S'
        else:
            if current_state[node] == 'S':
                if random.random() < PVACC:
                    next_state[node] = 'V'
                else:
                    for neighbor in G.neighbors(node):
                        if current_state[neighbor] == 'I':
                            if random.random() < ALPHA:
                                next_state[node] = 'I'
                            break
    return next_state

alpha = 0.1 
beta = 0.1    
gamma = 0.1  
pvacc = 0.1

sizes = [75, 75, 300]
probs = [[0.25, 0.05, 0.02], [0.05, 0.35, 0.07], [0.02, 0.07, 0.40]]
G = nx.stochastic_block_model(sizes, probs, seed=0)

#G.draw()
#G = nx.complete_graph(1000)

sim = Simulation(G, initial_state, state_transition_SIRV, alpha, beta, gamma, pvacc)
simulation_result = sim.run(10)

lista_nodi = list(G.nodes)
df_nodi = pd.DataFrame({'nodo': lista_nodi})
df_nodi.to_csv('nodi.csv', index=False)

lista_archi = list(G.edges)
df_archi = pd.DataFrame(lista_archi, columns=['nodo_origine', 'nodo_destinazione'])
df_archi.to_csv('archi.csv', index=False)

simulation_result.plot()

#simulation_result.draw()
    
plt.show()