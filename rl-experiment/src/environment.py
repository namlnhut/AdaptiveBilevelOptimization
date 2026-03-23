import numpy as np


class GridWorldEnvironment:
    """
    MDP environment for adaptive bilevel optimization experiments.

    Tensor conventions (matches main.py / main_euclidean.py):
        T  : ndarray, shape (n_actions, n_states, n_states)
             T[a, s, s'] = P(s' | s, a)
        r  : ndarray, shape (n_states, n_actions)
             r[s, a] = one-stage cost
        p_in: ndarray, shape (n_states,)
             initial-state distribution

    Parameters
    ----------
    grid_type : int
        0  – circular chain (each action shifts by a fixed number of steps)
        1  – random stochastic MDP (different seed)
        other – falls back to grid_type 0 logic with a different seed
    n_states : int
        Number of states.
    prop : float in [0, 1]
        Stochasticity level.  0 → deterministic transitions;
        1 → fully random (uniform mixture).
    """

    N_ACTIONS = 4  # fixed action space size for all grid types

    def __init__(self, grid_type: int, n_states: int, prop: float = 0.0):
        self.grid_type = grid_type
        self.n_states  = n_states
        self.n_actions = self.N_ACTIONS
        self.gamma     = 0.99   # default discount; overridden externally

        rng = np.random.default_rng(seed=grid_type * 137 + n_states)

        T = self._build_transitions(n_states, prop, rng, grid_type)
        self.T   = T                                              # (A, S, S)
        self.r   = rng.uniform(0.0, 1.0, (n_states, self.n_actions))  # (S, A)
        self.p_in = np.ones(n_states) / n_states                  # (S,)

    # ------------------------------------------------------------------
    def _build_transitions(self, n_states, prop, rng, grid_type):
        """
        Build T[a, s, s'] = P(s' | s, a).

        grid_type == 0 : circular chain
            action a moves the agent forward by (a+1) steps mod n_states.
        grid_type == 1 : random MDP
            base transitions are sampled from a Dirichlet distribution.
        otherwise      : circular chain (same as grid_type 0 but different seed
                         because of the rng initialisation).
        """
        A, S = self.n_actions, n_states
        T = np.zeros((A, S, S))

        if grid_type == 1:
            # Random base transitions (Dirichlet, concentrated on a few states)
            concentration = np.ones(S) * 0.5
            for a in range(A):
                for s in range(S):
                    T[a, s] = rng.dirichlet(concentration)
        else:
            # Circular chain: action a shifts state by (a + 1) steps
            for a in range(A):
                shift = a + 1
                for s in range(S):
                    T[a, s, (s + shift) % S] = 1.0

        if prop > 0.0:
            # Mix deterministic/structured component with uniform noise
            uniform = np.ones((A, S, S)) / S
            T = (1.0 - prop) * T + prop * uniform
            # Re-normalise rows (should already sum to 1, but be safe)
            T /= T.sum(axis=2, keepdims=True)

        return T
