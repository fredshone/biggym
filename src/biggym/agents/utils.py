import numpy as np


def revert_state(state_num: int, step: int) -> np.array:
    return state_num - (3 * step)

def state_index(obs: np.array, step: int) -> int:
    # takes in the int of 0,1,2 from the state and then converts to extended temporal state idk if this is right
    state_num = int(obs["curr_state"])
    return state_num + (3 * step)

def solve_tabular_mdp(*args, **kwargs):
    return value_iteration(*args, **kwargs)

def policy_evaluation(p, r, pi, gamma=0.99, epsilon=1e-2, max_iter=100):
    n_s, n_a = p.shape[:2]

    if pi.shape != (n_s, n_a):
        new_pi = np.zeros((n_s, n_a))
        new_pi[np.arange(n_s), pi] = 1
        pi = new_pi

    v = np.zeros(n_s)

    if len(r.shape) == 3:
        r = r.mean(axis=2)

    for t in range(int(max_iter)):
        q = r + np.einsum('ijk, k -> ij', p, gamma * v)
        v_ = np.einsum('ij, ij -> i', pi, q)        # equivalent to np.sum(pi * q, axis=1)

        dists = np.abs(v_ - v)
        diff = dists.max()

        # Check if value function is epsilon-optimal
        if diff < epsilon:
            break

        v = v_

    return v



def value_iteration(*args, **kwargs):
    # Just forward arguments to extended value iteration (by default
    # no confidence bounds)
    pi, (_, q, _, _) = extended_value_iteration(*args, **kwargs)

    return pi, q

def extended_value_iteration(p, r, cb_p=None, cb_r=None, gamma=0.99, epsilon=1e-2, max_iter=100, tau=0.999):
    is_value_iteration = cb_p is None or cb_r is None

    n_s, n_a = p.shape[:2]

    v = np.zeros(n_s)
    pi = np.zeros(n_s, dtype=int)

    if len(r.shape) == 3:
        # If the reward depends on the next state, we just do a weighted
        # average w.r.t. the transition probabilities
        r = np.sum(r * p, axis=2)

    r_tilde = r
    if not is_value_iteration:
        r_tilde += cb_r

    pi = np.zeros((n_s), dtype=int)

    # Identity transition matrix (for Schweitzer transformation)
    I_tran = np.tile(np.eye(n_s), (n_a, 1, 1)).transpose((1, 0, 2))

    for t in range(int(max_iter)):
        # Compute p tilde
        if is_value_iteration:
            p_tilde = p
        else:
            p_s_order = np.argsort(-v)

            p_tilde = np.zeros_like(p)
            for s in range(n_s):
                for a in range(n_a):
                    p_tilde[s, a] = inner_maximization(p[s, a], cb_p[s, a], p_s_order)

        # Apply Schweitzer transformation
        p_tilde = tau * p_tilde + (1. - tau) * I_tran

        # Compute policy pi
        q = r_tilde + np.einsum('ijk, k -> ij', p_tilde, gamma * v)
        pi_ = np.argmax(q, axis=1)
        v_ = np.max(q, axis=1)

        diff = v_ - v

        udpate_idx = diff > 0
        pi[udpate_idx] = pi_[udpate_idx]

        # TODO: why is the proof-verified condition not working for discounted case?
        # Does the proof only work for average reward setup?
        # if np.abs(diff).max() - np.abs(diff).min() < epsilon:
        if np.abs(diff).max() < epsilon:
            break

        v = v_

    v *= tau

    if gamma == 1:
        v -= v[0]

    return pi, (v, q, p_tilde, r_tilde)




def inner_maximization(p_sa_hat, confidence_bound_p_sa, rank):
    '''
    Find the best local transition p(.|s, a) within the plausible set of transitions as bounded by the confidence bound for some state action pair.
    Arg:
        p_sa_hat : (n_states)-shaped float array. MLE estimate for p(.|s, a).
        confidence_bound_p_sa : scalar. The confidence bound for p(.|s, a) in L1-norm.
        rank : (n_states)-shaped int array. The sorted list of states in descending order of value.
    Return:
        (n_states)-shaped float array. The optimistic transition p(.|s, a).
    '''

    min_val = np.minimum(1, p_sa_hat[rank[0]] + confidence_bound_p_sa / 2)

    if min_val == 1.:
        p_sa = np.zeros_like(p_sa_hat)
        p_sa[rank[0]] = min_val

        return p_sa

    p_sa = p_sa_hat.copy()
    p_sa[rank[0]] = min_val

    rank_dup = list(rank)
    last = rank_dup.pop()

    # Reduce until it is a distribution (equal to one within numerical tolerance)
    p_sa_sum = p_sa.sum()
    while round(p_sa_sum, 9) > 1:
        p_sa[last] = max(0, 1 - p_sa_sum + p_sa[last])
        p_sa_sum = p_sa.sum()
        last = rank_dup.pop()

    return p_sa


def get_policy_average_reward(p, r, pi, epsilon=1e-2, max_iter=100):
    if len(pi.shape) == 1:
        pi_idx = pi
        pi = np.zeros((len(pi_idx), len(pi_idx)))
        pi[np.arange(len(pi_idx)), pi_idx] = 1

    mu_pi = get_steady_state_distribution(p, pi, epsilon=epsilon, max_iter=max_iter)
    avg_reward = np.einsum("i,ij,ij", mu_pi, pi, r)

    return avg_reward


def get_steady_state_distribution(p, pi, epsilon=1e-2, max_iter=100):
    n_s, _ = p.shape[:2]
    mu_pi = np.ones(n_s) / n_s      # Initialize as uniform distribution

    for _ in range(int(max_iter)):
        next_mu_pi = np.einsum("i,ij,ijk->k", mu_pi, pi, p)

        diff = np.abs(next_mu_pi - mu_pi).max()
        mu_pi = next_mu_pi

        if diff < epsilon:
            break

    # Normalize in case sum is not exactly 1
    mu_pi /= mu_pi.sum()

    return mu_pi