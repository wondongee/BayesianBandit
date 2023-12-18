import numpy as np
import pymc3 as pm


class Agent(object):
 
    def __init__(self, bandit, policy, prior=0):
        self.policy = policy
        self.k = bandit.k
        self.prior = prior
        self._value_estimates = prior*np.ones(self.k)
        self.action_attempts = np.zeros(self.k)
        self.t = 0
        self.last_action = None

    def __str__(self):
        return 'f/{}'.format(str(self.policy))

    def reset(self):
        self._value_estimates[:] = self.prior
        self.action_attempts[:] = 0
        self.last_action = None
        self.t = 0

    def choose(self):
        action = self.policy.choose(self)
        self.last_action = action
        return action

    def observe(self, reward):
        self.action_attempts[self.last_action] += 1
        g = 1 / self.action_attempts[self.last_action]        
        q = self._value_estimates[self.last_action]

        self._value_estimates[self.last_action] += g*(reward - q)
        self.t += 1

    @property
    def value_estimates(self):
        return self._value_estimates
    
class NormalAgent(Agent):
    def __init__(self, bandit, policy):
        super(NormalAgent, self).__init__(bandit, policy)
        self.sq_mean = None
        self.model = pm.Model()
        with self.model:
            self._prior = pm.Normal('prior', mu=np.zeros(self.k), sd=np.ones(self.k),
                                    shape=(self.k), transform=None)

    def __str__(self):    
        return 'Normal/{}'.format(str(self.policy))

    def reset(self):
        super(NormalAgent, self).reset()
        self.sq_mean = [np.array([]) for _ in range(self.k)]
        self._prior.distribution.mu = np.zeros(self.k)
        self._prior.distribution.sd = np.ones(self.k)
        
    def observe(self, reward):
        self.action_attempts[self.last_action] += 1
        self.sq_mean[self.last_action] = np.append(self.sq_mean[self.last_action], reward**2)

        # update the mean
        n = self.action_attempts[self.last_action]
        old_mean = self._prior.distribution.mu[self.last_action]
        self._prior.distribution.mu[self.last_action] = old_mean + (reward - old_mean) / n
        
        # update the variance
        if n > 7:
            sq_mean = np.mean(self.sq_mean[self.last_action])
            self._prior.distribution.sd[self.last_action] = np.sqrt(sq_mean - self._prior.distribution.mu[self.last_action]**2)
        if self.t % 50 == 0:
            pass
            
        self._value_estimates = np.random.normal(self._prior.distribution.mu, self._prior.distribution.sd)
        self.t += 1
        
class NormalAgent_disc1(Agent):
    def __init__(self, bandit, policy):
        super(NormalAgent_disc1, self).__init__(bandit, policy)
        self.sq_mean = None
        self.model = pm.Model()
        with self.model:
            self._prior = pm.Normal('prior', mu=np.zeros(self.k), sd=np.ones(self.k),
                                    shape=(self.k), transform=None)

    def __str__(self):    
        return 'Normal/{}'.format(str(self.policy))

    def reset(self):
        super(NormalAgent_disc1, self).reset()
        self.sq_mean = [np.array([]) for _ in range(self.k)]
        self._prior.distribution.mu = np.zeros(self.k)
        self._prior.distribution.sd = np.ones(self.k)
        
    def observe(self, reward):
        self.action_attempts[self.last_action] += 1
        self.sq_mean[self.last_action] = np.append(self.sq_mean[self.last_action], reward**2)

        # update the mean
        n = self.action_attempts[self.last_action]
        old_mean = self._prior.distribution.mu[self.last_action]
        self._prior.distribution.mu[self.last_action] = old_mean + (reward - old_mean) / n
        
        # update the variance
        if n > 7:
            sq_mean = np.mean(self.sq_mean[self.last_action])
            decay_rate = 1 / (0.5 * np.log(n))
            self._prior.distribution.sd[self.last_action] = np.sqrt((sq_mean - self._prior.distribution.mu[self.last_action]**2))
            self._value_estimates = np.random.normal(self._prior.distribution.mu, self._prior.distribution.sd * decay_rate)
            self.t += 1
            return
        
        self._value_estimates = np.random.normal(self._prior.distribution.mu, self._prior.distribution.sd)
        self.t += 1


class NormalAgent_disc2(Agent):
    def __init__(self, bandit, policy):
        super(NormalAgent_disc2, self).__init__(bandit, policy)
        self.sq_mean = None
        self.model = pm.Model()
        with self.model:
            self._prior = pm.Normal('prior', mu=np.zeros(self.k), sd=np.ones(self.k),
                                    shape=(self.k), transform=None)

    def __str__(self):    
        return 'Normal/{}'.format(str(self.policy))

    def reset(self):
        super(NormalAgent_disc2, self).reset()
        self.sq_mean = [np.array([]) for _ in range(self.k)]
        self._prior.distribution.mu = np.zeros(self.k)
        self._prior.distribution.sd = np.ones(self.k)
        
    def observe(self, reward):
        self.action_attempts[self.last_action] += 1
        self.sq_mean[self.last_action] = np.append(self.sq_mean[self.last_action], reward**2)

        # update the mean
        n = self.action_attempts[self.last_action]
        old_mean = self._prior.distribution.mu[self.last_action]
        self._prior.distribution.mu[self.last_action] = old_mean + (reward - old_mean) / n
        
        # update the variance
        if n > 7:
            sq_mean = np.mean(self.sq_mean[self.last_action])
            decay_rate = (0.5 * np.sqrt(np.log(self.t)/n))
            self._prior.distribution.sd[self.last_action] = np.sqrt((sq_mean - self._prior.distribution.mu[self.last_action]**2))
            self._value_estimates = np.random.normal(self._prior.distribution.mu, self._prior.distribution.sd * decay_rate)
            self.t += 1
            return    
            
        self._value_estimates = np.random.normal(self._prior.distribution.mu, self._prior.distribution.sd)
        self.t += 1

