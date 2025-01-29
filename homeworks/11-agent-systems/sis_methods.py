import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random


def generate_er_graph(N, p, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    G = nx.erdos_renyi_graph(N, p, seed=seed)
    return G


def generate_ba_graph(N, m, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    G = nx.barabasi_albert_graph(N, m, seed=seed)
    return G


def initialize_states(N, initial_infected_fraction, seed=None):
    """
    Initialize the SIS states of N nodes: S=0, I=1.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    states = np.zeros(N, dtype=int)  # all susceptible by default
    num_initial_infected = int(initial_infected_fraction * N)
    infected_nodes = set(random.sample(range(N), num_initial_infected))
    
    for node in infected_nodes:
        states[node] = 1
    
    return states


def update_sis_states(states, G, beta, gamma):
    """
    Perform one SIS update step (discrete time) on the current state configuration.

    Parameters:
    -----------
    states : Current states of each node (0 for S, 1 for I).
    G : The underlying graph (ER).
    beta : Infection rate per edge per time step.
    gamma : Recovery rate per infected node per time step.

    Returns:
    --------
    new_states : np.array
        Updated states after one time-step of SIS dynamics.
    """
    N = len(states)
    new_states = states.copy()
    
    for node in range(N):
        if states[node] == 1:
            # Attempt recovery
            if random.random() < gamma:
                new_states[node] = 0
        else:
            # Attempt infection from infected neighbors
            neighbors = list(G.neighbors(node))
            infected_neighbors = sum(states[nbr] for nbr in neighbors)
            if infected_neighbors > 0:
                # Probability node becomes infected = 1 - (1 - beta)^(infected_neighbors)
                prob_infection = 1 - (1 - beta)**infected_neighbors
                if random.random() < prob_infection:
                    new_states[node] = 1
    
    return new_states


def run_sis_simulation_er(
    N=1000, 
    p=0.01, 
    beta=0.10, 
    gamma=0.05, 
    initial_infected_fraction=0.01, 
    max_time=50, 
    seed=None
):
    """
    Run a full SIS simulation on an ER graph for a specified number of time steps.

    Returns:
    --------
    infected_fraction : np.array
        Time series of fraction of infected nodes at each time step.
    susceptible_fraction : np.array
        Time series of fraction of susceptible nodes at each time step.
    """
    G = generate_er_graph(N, p, seed=seed)

    states = initialize_states(N, initial_infected_fraction, seed=seed)

    infected_fraction = []
    susceptible_fraction = []

    for _ in range(max_time):
        infected_fraction.append(np.mean(states == 1))
        susceptible_fraction.append(np.mean(states == 0))
        
        # Single SIS update step
        states = update_sis_states(states, G, beta, gamma)
    
    return np.array(infected_fraction), np.array(susceptible_fraction)



def run_sis_simulation_ba(
    N=1000, 
    m=2, 
    beta=0.10, 
    gamma=0.05, 
    initial_infected_fraction=0.01, 
    max_time=50, 
    seed=None
):
    G = generate_ba_graph(N, m, seed=seed)
    states = initialize_states(N, initial_infected_fraction, seed=seed)
    
    infected_fraction = []
    susceptible_fraction = []

    for _ in range(max_time):
        infected_fraction.append(np.mean(states == 1))
        susceptible_fraction.append(np.mean(states == 0))
        
        states = update_sis_states(states, G, beta, gamma)
    
    return np.array(infected_fraction), np.array(susceptible_fraction)