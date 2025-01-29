import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random


def generate_ws_graph(N, k, p, seed=None):
    """
    Generate a Watts–Strogatz small-world graph with N nodes,
    each node connected to k nearest neighbors, and rewiring probability p.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    G = nx.watts_strogatz_graph(n=N, k=k, p=p, seed=seed)
    return G


def initialize_sir_states(N, initial_infected_fraction=0.01, seed=None):
    """
    Initialize an array of node states for SIR:
      - 0 = Susceptible
      - 1 = Infected
      - 2 = Recovered (unused at t=0)
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    states = np.zeros(N, dtype=int)  # all susceptible
    num_initial_infected = int(initial_infected_fraction * N)
    infected_nodes = set(random.sample(range(N), num_initial_infected))
    
    for i in infected_nodes:
        states[i] = 1  # infected
    
    return states


def update_sir_states(states, G, beta, gamma):
    """
    Perform one update step for the SIR model:
      - Infected can infect their susceptible neighbors with prob beta
      - Infected recover with prob gamma and move to state 2 (Recovered)
    """
    new_states = states.copy()
    N = len(states)
    
    for node in range(N):
        if states[node] == 1:
            # Infect neighbors
            for nbr in G.neighbors(node):
                if states[nbr] == 0:  # Susceptible
                    if random.random() < beta:
                        new_states[nbr] = 1
            # Recover
            if random.random() < gamma:
                new_states[node] = 2  # Recovered
    
    return new_states


def run_sir_simulation_ws(
    N=1000,
    k=6,
    p=0.01,
    beta=0.10,
    gamma=0.05,
    initial_infected_fraction=0.01,
    max_time=50,
    seed=None
):
    """
    Run an SIR simulation on a Watts–Strogatz small-world network.
    Returns time series of fraction S, I, and R.
    """
    # 1. Generate WS graph
    G = generate_ws_graph(N, k, p, seed=seed)

    # 2. Initialize states
    states = initialize_sir_states(N, initial_infected_fraction, seed=seed)

    # 3. Arrays to keep track of S, I, R fractions
    S_frac, I_frac, R_frac = [], [], []

    # 4. Simulation loop
    for _ in range(max_time):
        S_frac.append(np.mean(states == 0))
        I_frac.append(np.mean(states == 1))
        R_frac.append(np.mean(states == 2))
        
        states = update_sir_states(states, G, beta, gamma)
    
    return np.array(S_frac), np.array(I_frac), np.array(R_frac)


