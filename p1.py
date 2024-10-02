import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

def prob_flip(Estado, i, j, beta):
    assert (i != len(Estado[0, :]) - 1) and (j != len(Estado[0, :]) - 1) and (i*j != 0), "No se puede cambiar un borde."
    valor = Estado[i, j]
    vecinos = Estado[[i-1, i, i, i+1], [j, j-1, j+1, j]]
    prob = np.exp(-2*beta*np.sum(valor*vecinos))
    normalizing_constant = np.sum([prob, np.exp(beta*np.sum(valor*vecinos))])
    return prob/normalizing_constant

def isingMH(N, beta, nf, x0):
    assert np.any(x0[0, :]) and np.any(x0[-1, :]) and np.any(x0[:, 0]) and np.any(x0[:, -1]), "La condición inicial debe tener borde 1."
    state = x0
    for _ in trange(nf):
        i, j = np.random.randint(1, N-1, 2)
        prob = prob_flip(state, i, j, beta)
        changed = np.random.binomial(1, prob)
        state[i, j] = -state[i, j] if changed else state[i, j]
    return state

if __name__ == "__main__":
    N = 10
    beta = 1.0
    nf = 200_000
    x0 = np.ones((N, N))
    state = isingMH(N, beta, nf, x0)
    plt.imshow(state)
    plt.show()
    print(state)



