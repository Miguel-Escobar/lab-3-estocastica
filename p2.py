import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm

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
    return array_caminos, [integral_de_camino(camino, posiciones, alpha) for camino in tqdm(array_caminos)]


if __name__ == "__main__":
    N = 20
    alpha = 1.0
    nf = 10_000
    constante = 1.0#((nf - 1) * 4 * np.sqrt(2) ** alpha)
    posiciones = np.random.rand(N, 2)
    caminos, integrales = vendedorViajeroSA(constante, posiciones, alpha, nf)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    beta_sucesión = np.log(np.arange(1, nf) + np.exp(1))/constante
    line_integral = axes[0].plot(np.arange(1), integrales[:1])
    axes[0].set_xlabel("Iteración")
    axes[0].set_ylabel(r"$\omega_\alpha$")
    beta_text = axes[0].text(0.75, 0.9, f"$\\beta = {beta_sucesión[0]:.2f}$", transform=axes[0].transAxes, fontsize=12)
    line_camino = axes[1].plot(np.tile(posiciones[caminos[0], 0], 2)[:(N+1)], np.tile(posiciones[caminos[0], 1], 2)[:(N+1)], "o-")
    axes[1].set_title("Camino actual")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")

    from matplotlib.animation import FuncAnimation

    def update(i):
        line_integral[0].set_xdata(np.arange(i))
        line_integral[0].set_ydata(integrales[:i])
        line_camino[0].set_data(np.tile(posiciones[caminos[i], 0], 2)[:(N+1)], np.tile(posiciones[caminos[i], 1], 2)[:(N+1)])
        axes[0].set_xlim(0, i)
        axes[0].set_ylim(np.min(integrales[:i])*0.9, np.max(integrales[:i])*1.1)
        beta_text.set_text(f"$\\beta = {beta_sucesión[i]:.2f}$")
        return line_integral[0], line_camino[0], beta_text
    
    anim = FuncAnimation(fig, update, frames=range(1, nf), interval = 10, blit=False)
    plt.show()