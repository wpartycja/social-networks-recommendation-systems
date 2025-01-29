import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random


def generate_random_network(N=50, p=0.1, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    G = nx.erdos_renyi_graph(N, p, seed=seed)
    return G

def initialize_opinions(N, seed=None):
    """
    Initialize each node's opinion to either +1 or -1 with equal probability.
    Returns a dict: node -> opinion
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    opinions = {}
    for node in range(N):
        opinions[node] = random.choice([-1, +1])
    return opinions

def initialize_stubbornness(N, seed=None):
    """
    Each agent has a 'stubbornness' parameter kappa_i in [0,1].
    - 0 means no stubbornness (always copies a disagreeing neighbor).
    - 1 means infinite stubbornness (never copies).
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    kappa = {}
    for node in range(N):
        kappa[node] = random.random()  # uniform in [0, 1]
    return kappa

def voter_update_step(G, opinions, kappa, alpha=0.05):
    """
    Perform one update of the 'modified voter model':
    
    - Pick random agent i.
    - With probability alpha, i adopts opinion = +1 (propaganda).
    - Else, pick random neighbor j of i:
      If opinions differ, i flips to j's opinion with probability (1 - kappa_i).
      If opinions are same, no change.
    """
    # 1. Pick a random agent
    i = random.choice(list(G.nodes()))

    # 2. Propaganda effect
    if random.random() < alpha:
        opinions[i] = +1
        return opinions

    # 3. If not propaganda, pick a neighbor
    neighbors = list(G.neighbors(i))
    if len(neighbors) == 0:
        return opinions  # no update if agent is isolated

    j = random.choice(neighbors)

    # 4. Compare opinions
    if opinions[i] != opinions[j]:
        # Probability that i flips is (1 - kappa_i)
        flip_prob = 1.0 - kappa[i]
        if random.random() < flip_prob:
            opinions[i] = opinions[j]
    
    return opinions

def run_modified_voter_model(
    N=50, 
    p=0.1, 
    alpha=0.05, 
    max_steps=1000, 
    seed=None
):
    G = generate_random_network(N, p, seed=seed)
    
    opinions = initialize_opinions(N, seed=seed)
    kappa = initialize_stubbornness(N, seed=seed)
    
    opinions_over_time = []
    
    for _ in range(max_steps):
        opinions_over_time.append(opinions.copy())
        
        opinions = voter_update_step(G, opinions, kappa, alpha=alpha)
    
    return opinions_over_time

def plot_opinion_evolution(opinions_over_time):
    plus_one_fraction = []
    minus_one_fraction = []
    
    for snapshot in opinions_over_time:
        all_ops = list(snapshot.values())
        count_plus = sum(1 for op in all_ops if op == +1)
        count_minus = sum(1 for op in all_ops if op == -1)
        total = len(all_ops)
        plus_one_fraction.append(count_plus / total)
        minus_one_fraction.append(count_minus / total)
    
    plt.figure(figsize=(8,5))
    plt.plot(plus_one_fraction, label='Opinion +1 fraction', color='red')
    plt.plot(minus_one_fraction, label='Opinion -1 fraction', color='blue')
    plt.xlabel("Step")
    plt.ylabel("Fraction of Agents")
    plt.legend()
    plt.grid(True)
    plt.title("Modified Voter Model with Propaganda & Stubbornness")
    plt.show()
