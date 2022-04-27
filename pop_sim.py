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

    # species_data is a dict of {name:{'r':r, 'relations':{target_species:val,}, 'reverse_relations':{target_species:val,}}}
    def __init__(self, species_data=None, h=0.00001):
        self.h = h
        if not species_data is None:
            assert isinstance(species_data, dict)

            self.species_names = sorted(list(species_data.keys()))
            self.species_relations = pd.DataFrame(0,index=self.species_names, columns=self.species_names)
            self.state_history = pd.DataFrame(columns=self.species_names)

            for name in species_data.keys():
                self.species_rs[name]=species_data[name]['r']
                for target in species_data[name]['relations'].keys():
                    self.species_relations.loc[name,target] = species_data[name]['relations'][target]
                for target in species_data[name]['reverse_relations'].keys():
                    self.species_relations.loc[target,name] = species_data[name]['reverse_relations'][target]

    #form A and r needed for integration
    def form_matrix(self,):
        assert (self.species_relations.index == self.species_relations.columns).all()
        self.A = self.species_relations.values
        self.r = [self.species_rs[name] for name in self.species_relations.index]

    # relations:{target_species:val,}, reverse_relations:{target_species:val,}
    def add_species(self, name, r, relations, reverse_relations):
        self.species_rs[name] = r
        self.state_history[name] = None

        if self.species_relations.empty:
            self.species_relations = pd.DataFrame(0, index=(name,), columns=(name,))
        else:
            self.species_relations.loc[name]=0
            self.species_relations.loc[:,name]=0

        for target in relations.keys():
            self.species_relations.loc[name,target] = relations[target]
        for target in reverse_relations.keys():
            self.species_relations.loc[target,name] = reverse_relations[target]
        self.species_names = self.species_relations.index

    # relations:{target_species:val,}, reverse_relations:{target_species:val,}
    # if clear is True it removes existing relations, else it just overwrites any given relations
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

        self.species_names = self.species_relations.index

    # if clear is True it sets any unspecified species to 0, else it just overwrites any given relations
    def set_state(self, state_dict, clear=False):
        if self.state_history.empty or clear:
            self.state_history = pd.DataFrame(state_dict, index=[0], columns=self.species_names)
        else:
            t = self.state_history.index[-1] + self.h
            self.state_history.loc[t,:] = self.state_history.iloc[-1,:]
            for name in state_dict.keys():
                self.state_history.loc[t,name] = state_dict[name]

    # the partial derivatives used for integration
    def dx_dt(self, state, t):
        state = np.reshape(np.asarray(state), (-1,1))
        f = np.reshape(self.r,(-1,1)) + np.matmul(self.A, state)
        dx_dt = np.multiply(state,f)
        dx_dt = tuple(np.reshape(dx_dt, (-1)))
        return dx_dt

    #integrate for time T building off of existing state
    def intergrate(self,T):
        self.form_matrix()
        h = self.h
        last_t = self.state_history.index[-1]
        last_state = self.state_history.iloc[-1,:].fillna(0).values
        t = np.arange(last_t+h, last_t+T+h, h)
        states = odeint(self.dx_dt, list(tuple(last_state)), t)

        self.state_history = self.state_history.append(pd.DataFrame(states, index=t, columns=self.state_history.columns))

    #Plot the entire state history, need to expand on this
    def plot(self,):
        for name in self.species_names:
            plt.plot(self.state_history.index, self.state_history[name], label=name)
        # plt.plot(states[:,1])
        plt.legend(loc='upper right')
        
        title = self.species_names[0]
        for species in self.species_names[1:]:
            title += f' vs. {species} '
        plt.title(title)
        plt.xlabel('time (t)')
        plt.ylabel('Population Size')
        plt.grid(True, linewidth=0.5)
        plt.show()
        # plt.plot(states[:,0],states[:,1])
        # plt.show()


    #String representation for debuging purpuses
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

def example():
    # species_data = {'spec1':{'r':-0.4, 'relations':{'spec2':0.3,'spec3':0.1}, 'reverse_relations':{}},
    #                 'spec2':{'r':0.2, 'relations':{'spec1':-0.3,'spec3':0.1}, 'reverse_relations':{}},
    #                 'spec3':{'r':0.2, 'relations':{'spec1':-0.1,'spec2':-0.1}, 'reverse_relations':{}},}
    species_data = {'spec1':{'r':-0.15, 'relations':{'spec2':0,'spec3':0.15}, 'reverse_relations':{}},
                    'spec2':{'r':-0.1, 'relations':{'spec1':0,'spec3':0.1}, 'reverse_relations':{}},
                    'spec3':{'r':0.2, 'relations':{'spec1':-0.1,'spec2':-0.1}, 'reverse_relations':{}},}
    print(species_data)
    sim = GeneralizedLotkaVolterraSim(h=0.00005, species_data=species_data)
    sim.set_state({'spec1':1.0, 'spec2':1.0, 'spec3':2.0})
    sim.form_matrix()
    print(sim)
    # sim.add_species(name='Rabbit', r=1.0, relations={}, reverse_relations={})
    sim.intergrate(250.0)
    print(sim)
    sim.plot()



if __name__ == '__main__':
    # tests()
    example()
