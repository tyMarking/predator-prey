import numpy as np
import collections.abc
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class GeneralizedLotkaVolterraSim():

    state_history = pd.DataFrame()

    #order matters!
    species_names = []
    species_rs = {}
    species_relations = pd.DataFrame()
    A = None
    r = None
    sate = None
    h = None
    # species
    def __init__(self, species_data=None, h=0.00001):
        self.h = h

        if not species_data is None:
            assert isinstance(species_data, dict)

            self.species_names = sorted(list(species_data.keys()))

            self.species_relations = pd.DataFrame(0,index=self.species_names, columns=self.species_names)
            self.state_history = pd.DataFrame(columns=self.species_names)

            for name in species_data.keys():
                self.rs[name]=species_data[name]['r']
                for target in species_data[name['relations']].keys():
                    self.species_relations.loc[name,target] = species_data[name['relations']][target]
                for target in species_data[name['reverse_relations']].keys():
                    self.species_relations.loc[target,name] = species_data[name['reverse_relations']][target]

    def form_matrix(self,):
        assert (self.species_relations.index == self.species_relations.columns).all()
        self.A = self.species_relations.values
        self.r = [self.species_rs[name] for name in self.species_relations.index]


    def add_species(self, name, r, relations, reverse_relations):
        self.species_rs[name] = r
        self.state_history[name] = None
        # self.species_relations.loc[name] =
        if self.species_relations.empty:
            self.species_relations = pd.DataFrame(0, index=(name,), columns=(name,))
        else:
            self.species_relations.loc[name]=0
            self.species_relations.loc[:,name]=0

        for target in relations.keys():
            self.species_relations.loc[name,target] = relations[target]
        for target in reverse_relations.keys():
            self.species_relations.loc[target,name] = reverse_relations[target]
        self.form_matrix()
        self.species_names = self.species_relations.index


    def update_species(self, name, r=None, relations=None, reverse_relations=None, clear=False):
        if not(r is None):
            self.species_rs[name] = r
        if not(relations is None):
            if clear:
                self.species_relations.loc[name,:] = 0
            for target in relations.keys():
                self.species_relations.loc[name,target] = relations[target]
        if not(reverse_relations is None):
            if clear:
                self.species_relations.loc[:,name] = 0
            for target in reverse_relations.keys():
                self.species_relations.loc[target,name] = reverse_relations[target]

        self.form_matrix()
        self.species_names = self.species_relations.index
    def set_state(self, state_dict, clear=True):
        # state0 = pd.Series(0,index=self.species_relations.index)
        # for name in state_dict.keys():
        #     state0.loc[name] = state_dict[name]
        if self.state_history.empty:
            t=0
        else:
            t = self.state_history.index[-1] + self.h
        if clear:
            self.state_history = self.state_history.append(pd.DataFrame(state_dict,index=[t]))
        else:
            self.state_history.loc[t,:] = self.state_history.iloc[-1,:]
            for name in state_dict.keys():
                self.state_history.loc[t,name] = state_dict[name]
    def dx_dt(self, state, t):
        state = np.reshape(np.asarray(state), (-1,1))
        f = np.reshape(self.r,(-1,1)) + np.matmul(self.A, state)
        dx_dt = np.multiply(state,f)
        dx_dt = tuple(np.reshape(dx_dt, (-1)))
        # print(dx_dt)
        return dx_dt

    def intergrate(self,T):
        h = self.h
        last_t = self.state_history.index[-1]
        print(self.state_history.tail(1))
        last_state = self.state_history.iloc[-1,:].fillna(0).values
        print(last_state)
        t = np.arange(last_t+h, last_t+T+h, h)
        states = odeint(self.dx_dt, list(tuple(last_state)), t)
        print(states)
        # plt.plot(states[:,0],states[:,1])
        # plt.show()
        self.state_history = self.state_history.append(pd.DataFrame(states, index=t, columns=self.state_history.columns))


    def plot(self,):
        for name in self.species_names:
            plt.plot(self.state_history.index, self.state_history[name], label=name)
        # plt.plot(states[:,1])
        plt.legend(loc='upper left')
        plt.show()

    def __str__(self,):
        ret = ''
        ret += "Species names:\n" + str(self.species_names) + "\n\n"
        ret += "Species rs:\n" + str(self.species_rs) + "\n\n"
        ret += "Species relations:\n" + str(self.species_relations) + "\n\n"
        ret += "State history:\n" + str(self.state_history) + "\n\n"
        ret += "A:\n" + str(self.A) + "\n"
        ret += "r:\n" + str(self.r) + "\n"
        ret += "h:\n" + str(self.h) + "\n"
        return ret


def tests():
    sim = GeneralizedLotkaVolterraSim(h=0.00001)
    print(sim)
    sim.add_species(name='Rabbit', r=1.0, relations={}, reverse_relations={})
    print(sim)
    sim.add_species(name='Fox', r=-2.0/3.0, relations={'Rabbit':4.0/3.0}, reverse_relations={'Rabbit':-1.0})
    print(sim)
    sim.form_matrix()
    print(sim)
    sim.set_state({'Rabbit':1, 'Fox':10})
    print(sim)
    sim.intergrate(100.0)
    print(sim)
    sim.plot()
    sim.add_species(name='Grass', r=0.5, relations={'Rabbit': -1.0}, reverse_relations={'Rabbit':2.0})
    sim.update_species(name='Rabbit', r=-1.0)
    print(sim)
    sim.set_state({'Grass':1}, clear=False)
    print(sim)
    sim.intergrate(100.0)
    print(sim)
    sim.plot()

if __name__ == '__main__':
    tests()
