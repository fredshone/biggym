import numpy as np
import pickle
# import jax
# import jax.numpy as jnp
import jax.random as jrandom
import distrax

from .utils import solve_tabular_mdp
import sys


class PSRLAgent:
    def __init__(self, env, config, key):
        self.env = env
        self.config = config
        self.epsilon = 1

        mu = config.MU
        lambd = config.LAMBDA
        alpha = config.ALPHA
        beta = config.BETA

        n_s = env.observation_space.n
        n_a = env.action_space.n

        # Initialize posterior distributions
        self.p_dist = config.KAPPA * np.ones((n_s, n_a, n_s))

        self.r_dist = (
            np.ones((n_s, n_a)) * mu,
            np.ones((n_s, n_a)) * lambd,
            np.ones((n_s, n_a)) * alpha,
            np.ones((n_s, n_a)) * beta,
        )

        self.pi = None
        self.p_count = np.zeros((n_s, n_a, n_s)).tolist()
        self.r_total = np.zeros((n_s, n_a)).tolist()
        self.steps = 0

        self.update_policy(key)

    def reset(self, state):
        pass

    def act(self, state, step, key: jrandom.PRNGKey):
        self.steps += 1

        # if self.steps % self.config.tau == 0:
        self.update_policy(key)

        return self.pi[state]

    def observe(self, transition):
        s, a, r, s_ = transition

        self.p_count[s][a][s_] += 1
        self.r_total[s][a] += r

    def update(self, obs, action, reward, nobs, step, key: jrandom.PRNGKey):
        if self.steps % self.config.TAU == 0:
            self.update_posterior()
            self.update_policy(key)

    def update_posterior(self):
        n_s = self.env.observation_space.n
        n_a = self.env.action_space.n

        p_count = np.asarray(self.p_count)
        r_total = np.asarray(self.r_total)

        # Update transition probabilities
        self.p_dist += p_count

        # Update reward function
        mu0, lambd, alpha, beta = self.r_dist

        r_count = p_count.sum(axis=2)
        mu = (lambd * mu0 + r_total) / (lambd + r_count)
        lambd += r_count
        alpha += r_count / 2.
        beta += (r_total ** 2. + lambd * mu0 ** 2. - lambd * mu ** 2.) / 2

        self.r_dist = (mu, lambd, alpha, beta)

        if self.steps % self.config.TAU == 0:
            self.p_count = np.zeros((n_s, n_a, n_s)).tolist()
            self.r_total = np.zeros((n_s, n_a)).tolist()

    def update_policy(self, key: jrandom.PRNGKey):
        ### Sample from posterior
        # Compute transition probabilities
        p = distrax.Dirichlet(self.p_dist).sample(seed=key)

        # Compute reward function
        mu0, lambd, alpha, beta = self.r_dist

        tau = distrax.Gamma(alpha, 1. / beta).sample(seed=key)
        mu = distrax.Normal(mu0, 1. / np.sqrt(lambd * tau)).sample(seed=key)

        r = mu

        ### Solve for optimal policy
        # self.pi, _ = solve_tabular_mdp(
        #     p.numpy(),
        #     r.numpy(),
        #     gamma=self.config.GAMMA,
        #     max_iter=self.config.MAX_ITER
        # )
        self.pi, _ = solve_tabular_mdp(
            p,
            r,
            gamma=self.config.GAMMA,
            max_iter=self.config.MAX_ITER
        )

    # def save(self, path):
    #     data = {
    #         'p_dist': self.p_dist,
    #         'r_dist': self.r_dist,
    #     }
    #
    #     with open(path, 'wb') as out_file:
    #         pickle.dump(data, out_file)
    #
    # def load(self, path):
    #     with open(path, 'rb') as in_file:
    #         data = pickle.load(in_file)
    #
    #     self.p_dist = data['p_dist']
    #     self.r_dist = data['r_dist']