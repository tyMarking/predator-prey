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
        # assert (species_names is None) == (species_r is None) == (A is None), "You must define either none or all of names, r, and A"
        # assert (state0 is None) or (not species_names is None), "Species must be defined to set an initial state"
        # if (not species_names is None) and (not species_r is None):
        #     assert len(species_names) == len(species_r), "Length of Species Name and r must be the same"
        # if (not species_names is None) and (not A is None):
        #     assert np.asarray(A).shape == (len(species_names), len(species_names)), f"A has shape {np.asarray(A).shape} when it should be {(len(species_names), len(species_names))}"
        # if not state0 is None:
        #     assert len(state0) == len(species_names), "Length of Species Name and initial state must be the same"
        #
        # if species_names is None:
        #     self.names = np.array([])
        # else:
        #     assert (isinstance(species_names, collections.abc.Sequence) and not isinstance(species_names, str)) or isinstance(species_names, np.ndarray), f"Type of species names cannot be {type(species_names)}, it must be a sequence"
        #     self.names = np.asarray(species_names)
        #
        # if species_r is None:
        #     self.r = np.array([])
        # else:
        #     assert (isinstance(species_r, collections.abc.Sequence) and not isinstance(species_r, str)) or isinstance(species_r, np.ndarray), f"Type of species r cannot be {type(species_r)}, it must be a sequence"
        #     self.r = np.reshape(np.asarray(species_r), (-1,1))
        #
        # if A is None:
        #     self.A = np.array([])
        # else:
        #     assert (isinstance(A, collections.abc.Sequence) and not isinstance(A, str)) or isinstance(A, np.ndarray), f"Type of A cannot be {type(A)}, it must be a sequence"
        #     self.A = np.asarray(A)
        #
        # if state0 is None:
        #     self.state0 = np.array([])
        # else:
        #     assert (isinstance(state0, collections.abc.Sequence) and not isinstance(state0, str)) or isinstance(state0, np.ndarray), f"Type of state0 cannot be {type(state0)}, it must be a sequence"
        #     self.state0 = np.asarray(state0)
        #     self.current_state = state0
        #     self.state_history.loc[0] = state0
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

    def set_state(self, state_dict):
        # state0 = pd.Series(0,index=self.species_relations.index)
        # for name in state_dict.keys():
        #     state0.loc[name] = state_dict[name]
        if self.state_history.empty:
            t=0
        else:
            t = self.state_history.index[-1] + self.h

        self.state_history = self.state_history.append(pd.DataFrame(state_dict,index=[t]))

    def dx_dt(self, state, t):
        state = np.reshape(np.asarray(state), (-1,1))
        f = np.reshape(self.r,(-1,1)) + np.matmul(self.A, state)
        dx_dt = np.multiply(state,f)
        dx_dt = tuple(np.reshape(dx_dt, (-1)))
        # print(dx_dt)
        return dx_dt

    def intergrate(self,h,T):
        last_t = self.state_history.index[-1]
        print(self.state_history.tail(1))
        last_state = self.state_history.iloc[-1,:].fillna(0).values
        print(last_state)
        t = np.arange(last_t+h, T+h, h)
        states = odeint(self.dx_dt, list(tuple(last_state)), t)
        print(states)
        plt.plot(states[:,0],states[:,1])
        plt.show()

        plt.plot(states[:,0])
        plt.plot(states[:,1])
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

            # state_history = pd.DataFrame()
            # A = None
            # r = None
        return ret

def tests():
    sim = GeneralizedLotkaVolterraSim()
    # sim = GeneralizedLotkaVolterraSim(species_names=['Rabbit', 'Fox'])
    # sim = GeneralizedLotkaVolterraSim(species_names='this should fail')

    # sim = GeneralizedLotkaVolterraSim(species_names=['Rabbit', 'Fox'], species_r=[1.0])
    # sim = GeneralizedLotkaVolterraSim(species_names=['Rabbit', 'Fox'], species_r='ab')
    # sim = GeneralizedLotkaVolterraSim(species_names=['Rabbit', 'Fox'], species_r=[1.0, 2.0])

    # sim = GeneralizedLotkaVolterraSim(species_names=['Rabbit', 'Fox'], A=[[1.0, 2.0],[0.0,0.1]])
    # sim = GeneralizedLotkaVolterraSim(species_names=['Rabbit', 'Fox'], A=[[1.0],[0.1]])

    sim = GeneralizedLotkaVolterraSim(species_names=['Rabbit', 'Fox'], species_r=[1.0, 2.0], A=[[1.0, 2.0],[0.0,0.1]])
    sim = GeneralizedLotkaVolterraSim(species_names=['Rabbit', 'Fox'], species_r=[1.0, 2.0], A=np.asarray([[1.0, 2.0],[0.0,0.1]]))

    sim = GeneralizedLotkaVolterraSim(species_names=['Fox', 'Rabbit'], species_r=[2.0/3.0, -1.0], A=np.asarray([[0.0, -4.0/3.0],[1.0,0.0]]), state0=[1.0,10.0])

    # sim.dx_dt(sim.state0, 0)

    sim.intergrate(0.00001, 100.0)
    # print("empty")
    # sim = GeneralizedLotkaVolterraSim()

def tests2():
    sim = GeneralizedLotkaVolterraSim()
    print(sim)
    sim.add_species(name='Rabbit', r=-1.0, relations={}, reverse_relations={})
    print(sim)
    sim.add_species(name='Fox', r=2.0/3.0, relations={'Rabbit':-4.0/3.0}, reverse_relations={'Rabbit':1.0})
    print(sim)
    sim.form_matrix()
    print(sim)
    sim.set_state({'Rabbit':1, 'Fox':10})
    print(sim)
    sim.intergrate(0.00001, 100.0)

if __name__ == '__main__':
    tests2()
