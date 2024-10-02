import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

def integral_de_camino(camino, posiciones, alpha):
    posiciones_ordenadas = posiciones[camino]
    return np.sum(np.linalg.norm(posiciones_ordenadas - np.roll(posiciones_ordenadas, -1, axis=0), axis=1)**alpha)

def prob_permutar(camino, i, j, posiciones, beta, alpha):
    assert (i*j != 0), "No se puede cambiar el punto de partida/llegada."
    camino_permutado = camino.copy()
    camino_permutado[[i, j]] = camino_permutado[[j, i]]
    prob_permutar = np.exp(-beta*(integral_de_camino(camino_permutado, posiciones, alpha) - integral_de_camino(camino, posiciones, alpha)))
    return prob_permutar

def vendedorViajeroSA(constante, posiciones, alpha, nf):
    N = len(posiciones[:, 0])
    array_caminos = np.zeros((nf, N), dtype=int)
    array_caminos[0, :] = np.arange(N) # sería nuestro sigma. Inicializamos con la identidad. No cambiamos ni la primera ni la última ciudad.
    beta_sucesión = np.log(np.arange(1, nf) + np.exp(1))/constante
    for i in trange(nf -1):
        beta = beta_sucesión[i]
        camino = array_caminos[i, :]
        ciudad_i, ciudad_j = np.random.randint(1, N, 2)
        prob = prob_permutar(camino, ciudad_i, ciudad_j, posiciones, beta, alpha)
        if np.random.rand() <= prob:
            camino[[ciudad_i, ciudad_j]] = camino[[ciudad_j, ciudad_i]]
        array_caminos[i+1, :] = camino
    return array_caminos, [integral_de_camino(camino, posiciones, alpha) for camino in array_caminos]

if __name__ == "__main__":
    N = 100
    alpha = 1.0
    nf = 1_000_000
    constante = 1.0
    posiciones = np.random.rand(N, 2)
    caminos, integrales = vendedorViajeroSA(constante, posiciones, alpha, nf)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].plot(integrales)
    axes[0].set_xlabel("Iteración")
    axes[0].set_ylabel(r"$\omega_\alpha$")
    axes[1].hist(integrales, bins=40, density=True)
    axes[1].set_xlabel(r"$\omega_\alpha$")
    axes[1].set_ylabel("Densidad")
    fig.show()