import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

ALPHA = 2.0/3.0
BETA = 4.0/3.0
DELTA = 1.0
GAMMA = 1.0

# ALPHA = 1.1
# BETA = 0.4
# # DELTA = 0.4
# DELTA = 0.1
# # GAMMA = 0.1
# GAMMA = 0.4


def lotka_volterra(state,t):
    x, y = state
    return (ALPHA*x)-(BETA*x*y), (DELTA*x*y) - (GAMMA*y)


def main():
    state0 = [1,10]
    t = np.arange(0.0, 100.0, 0.00001)
    states = odeint(lotka_volterra, state0, t)
    print(states)
    plt.plot(states[:,0],states[:,1])
    plt.show()

    plt.plot(states[:,0])
    plt.plot(states[:,1])
    plt.show()


if __name__ == '__main__':
    main()
