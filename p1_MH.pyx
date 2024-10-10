import numpy as np
cimport numpy as np

# prob_flip function
def prob_flip(np.ndarray[np.float64_t, ndim=2] Estado, int i, int j) -> float:
    cdef double valor = Estado[i, j]
    cdef double neighbors_sum = (Estado[i-1, j] + Estado[i+1, j] + Estado[i, j+1] + Estado[i, j-1])
    cdef double prob = valor * neighbors_sum
    return prob

# isingMH function
def isingMH(int N, float beta, int nf, np.ndarray[np.float64_t, ndim=2] x0):
    # Check for boundary condition
    assert np.all(x0[0, :]) and np.all(x0[-1, :]) and np.all(x0[:, 0]) and np.all(x0[:, -1]), "Initial condition must have boundary 1."
    
    cdef np.ndarray[np.float64_t, ndim=2] state = x0.copy()
    cdef np.ndarray[np.int_t, ndim=2] random_indices = np.random.randint(1, N-1, (nf, 2))
    cdef np.ndarray[np.float64_t, ndim=1] log_uniforms = np.log(np.random.rand(nf)) / (-2 * beta)
    
    cdef int i, j
    cdef double r, prob
    for idx in range(nf):
        i, j = random_indices[idx, 0], random_indices[idx, 1]
        prob = prob_flip(state, i, j)
        r = log_uniforms[idx]
        if r >= prob:
            state[i, j] *= -1
    
    return state
