import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats

class Environment(object):
    def __init__(self, bandit, agents, label='Multi-Armed Bandit'):
        self.bandit = bandit
        self.agents = agents
        self.label = label

    def reset(self):
        self.bandit.reset()
        for agent in self.agents:
            agent.reset()

    def run(self, trials=100, experiments=1):
        scores = np.zeros((trials, len(self.agents)))
        optimal = np.zeros_like(scores)

        for _ in range(experiments):
            self.reset()
            for t in range(trials):
                for i, agent in enumerate(self.agents):
                    action = agent.choose()
                    reward, is_optimal = self.bandit.pull(action)
                    agent.observe(reward)

                    scores[t, i] += reward
                    if is_optimal:
                        optimal[t, i] += 1

        return scores / experiments, optimal / experiments

    def plot_results(self, scores, optimal):
        sns.set_style('white')
        sns.set_context('talk')
        
        plt.figure(figsize=(16, 16))
        
        # First subplot
        plt.subplot(2, 1, 1)
        plt.title(self.label)
        plt.plot(scores, lw=2.0)
        plt.ylabel('Average Reward')
        plt.legend(self.agents, loc=4)
        
        # Second subplot
        plt.subplot(2, 1, 2)
        plt.plot(optimal * 100, lw=2.0)
        plt.ylim(0, 100)
        plt.ylabel('% Optimal Action')
        plt.xlabel('Time Step')
        plt.legend(self.agents, loc=4)        
        
        sns.despine()
        plt.show()
