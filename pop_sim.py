import numpy as np
import collections.abc
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class GeneralizedLotkaVolterraSim():

    state_history = pd.Series()
    current_state = None

    def __init__(self, species_names=None, species_r=None, A=None, state0=None):
        assert (species_names is None) == (species_r is None) == (A is None), "You must define either none or all of names, r, and A"
        assert (state0 is None) or (not species_names is None), "Species must be defined to set an initial state"
        if (not species_names is None) and (not species_r is None):
            assert len(species_names) == len(species_r), "Length of Species Name and r must be the same"
        if (not species_names is None) and (not A is None):
            assert np.asarray(A).shape == (len(species_names), len(species_names)), f"A has shape {np.asarray(A).shape} when it should be {(len(species_names), len(species_names))}"
        if not state0 is None:
            assert len(state0) == len(species_names), "Length of Species Name and initial state must be the same"

        if species_names is None:
            self.names = np.array([])
        else:
            assert (isinstance(species_names, collections.abc.Sequence) and not isinstance(species_names, str)) or isinstance(species_names, np.ndarray), f"Type of species names cannot be {type(species_names)}, it must be a sequence"
            self.names = np.asarray(species_names)

        if species_r is None:
            self.r = np.array([])
        else:
            assert (isinstance(species_r, collections.abc.Sequence) and not isinstance(species_r, str)) or isinstance(species_r, np.ndarray), f"Type of species r cannot be {type(species_r)}, it must be a sequence"
            self.r = np.reshape(np.asarray(species_r), (-1,1))

        if A is None:
            self.A = np.array([])
        else:
            assert (isinstance(A, collections.abc.Sequence) and not isinstance(A, str)) or isinstance(A, np.ndarray), f"Type of A cannot be {type(A)}, it must be a sequence"
            self.A = np.asarray(A)

        if state0 is None:
            self.state0 = np.array([])
        else:
            assert (isinstance(state0, collections.abc.Sequence) and not isinstance(state0, str)) or isinstance(state0, np.ndarray), f"Type of state0 cannot be {type(state0)}, it must be a sequence"
            self.state0 = np.asarray(state0)
            self.current_state = state0
            self.state_history.loc[0] = state0


    #TODO
    # def add_species(self, name, r, relations):
        # assert len(relations == )

    def dx_dt(self, state, t):
        state = np.reshape(np.asarray(state), (-1,1))
        f = self.r + np.matmul(self.A, state)
        dx_dt = np.multiply(state,f)
        dx_dt = tuple(np.reshape(dx_dt, (-1)))
        # print(dx_dt)
        return dx_dt

    def intergrate(self,h,T):
        t = np.arange(0.0, T, h)
        states = odeint(self.dx_dt, list(tuple(self.state0)), t)
        print(states)
        plt.plot(states[:,0],states[:,1])
        plt.show()

        plt.plot(states[:,0])
        plt.plot(states[:,1])
        plt.show()

        
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

if __name__ == '__main__':
    tests()
