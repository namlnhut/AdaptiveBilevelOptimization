import numpy as np


class MDPsolver:
    """
    Exact MDP solver for a GridWorldEnvironment.

    Attributes
    ----------
    v  : ndarray, shape (n_states,)
         Optimal value function (set after value_iteration()).
    pi : ndarray, shape (n_states, n_actions)
         Optimal greedy policy (set after value_iteration()).
    """

    def __init__(self, env):
        self.env = env
        self.v  = np.zeros(env.n_states)
        self.pi = np.ones((env.n_states, env.n_actions)) / env.n_actions

    # ------------------------------------------------------------------
    def value_iteration(self, tol: float = 1e-12, max_iter: int = 100_000):
        """
        Solve for the optimal value function via synchronous value iteration.

        Updates self.v (optimal value) and self.pi (optimal greedy policy).
        Minimises expected cumulative discounted cost (self.env.r is a cost).
        """
        env   = self.env
        gamma = env.gamma
        v     = np.zeros(env.n_states)

        for _ in range(max_iter):
            # Q[s, a] = r[s, a] + gamma * sum_{s'} T[a, s, s'] * v[s']
            # einsum 'asj, j -> sa' : T (A,S,S') dot v (S') -> (S, A)
            Q     = env.r + gamma * np.einsum('asj,j->sa', env.T, v)
            new_v = Q.max(axis=1)   # maximise reward
            if np.max(np.abs(new_v - v)) < tol:
                v = new_v
                break
            v = new_v

        self.v = v

        # Greedy policy: deterministic, put all mass on the cheapest action
        Q        = env.r + gamma * np.einsum('asj,j->sa', env.T, v)
        best_a   = np.argmax(Q, axis=1)
        self.pi  = np.zeros_like(env.r)
        for s in range(env.n_states):
            self.pi[s, best_a[s]] = 1.0

    # ------------------------------------------------------------------
    def pi_eval(self, policy: np.ndarray) -> np.ndarray:
        """
        Exact policy evaluation via direct linear-system solve.

        Solves  (I - gamma * P_pi) v = r_pi  for v.

        Parameters
        ----------
        policy : ndarray, shape (n_states, n_actions)
            Stochastic policy; each row must sum to 1.

        Returns
        -------
        v : ndarray, shape (n_states,)
            State-value function of the given policy.
        """
        env   = self.env
        gamma = env.gamma
        S     = env.n_states

        # r_pi[s] = sum_a policy[s, a] * r[s, a]
        r_pi = np.sum(policy * env.r, axis=1)               # (S,)

        # P_pi[s, s'] = sum_a policy[s, a] * T[a, s, s']
        # einsum 'sa, asj -> sj'
        P_pi = np.einsum('sa,asj->sj', policy, env.T)       # (S, S)

        # Solve (I - gamma * P_pi) v = r_pi
        A_mat = np.eye(S) - gamma * P_pi
        v     = np.linalg.solve(A_mat, r_pi)
        return v
