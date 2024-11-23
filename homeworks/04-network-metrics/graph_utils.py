import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from scipy.stats import linregress


def draw_basic_histrogram(G):
    degrees = [degree for node, degree in G.degree()]

    plt.hist(degrees, bins=10, color='skyblue', edgecolor='black') 
    plt.xlabel("Number of Connections (Degree)")
    plt.ylabel("Frequency")
    plt.show()
    
    
def draw_log_scale_histogram(G, num_bins=10,log_bin=False):
    degrees = [degree for node, degree in G.degree()]

    
    if log_bin:
        min_degree = min(degrees)
        max_degree = max(degrees)

        bins = np.logspace(np.log10(min_degree), np.log10(max_degree), num_bins)

        plt.hist(degrees, bins=bins, color='skyblue', edgecolor='black')

        plt.xscale("log")
        plt.yscale("log")

        plt.xlabel("Number of Connections (Degree, Log Scale)")
        plt.ylabel("Frequency (Log Scale)")

        plt.show()
    else:
        plt.hist(degrees, bins=num_bins, color='skyblue', edgecolor='black', log=True)  
        
        plt.xscale("log")
        plt.yscale("log")
        
        plt.xlabel("Number of Connections (Degree)")
        plt.ylabel("Frequency (Log Scale)")
        plt.show()


def draw_survival_func(G, reg=False):
    degrees = [degree for node, degree in G.degree()]
        
    degree_counts = Counter(degrees)
    degree_values = np.array(sorted(degree_counts.keys()))
    degree_frequencies = np.array([degree_counts[d] for d in degree_values])

    cdf = np.cumsum(degree_frequencies) / len(degrees)
    survival_function = 1 - cdf

    plt.plot(degree_values, survival_function, marker="o",  color="skyblue")
    plt.title("Survival Function")
    plt.xlabel("Number of Connections (Degree)")
    plt.ylabel("Survival Probability (1 - CDF)")
    
    plt.xscale("log")
    plt.yscale("log")
    
    if reg:
        survival_function = np.clip(survival_function, a_min=1e-10, a_max=None)
        degree_values = np.clip(degree_values, a_min=1e-10, a_max=None)
        
        log_degree_values = np.log(degree_values)
        log_survival_function = np.log(survival_function)

        slope, intercept, r_value, p_value, std_err = linregress(log_degree_values, log_survival_function)
        plt.plot(log_degree_values, intercept + slope * log_degree_values, 'r', label=f'Linear Fit: slope = {slope:.2f}')

    plt.show()
    
def calculate_alpha(G):
    degrees = [degree for node, degree in G.degree()]
        
    degree_counts = Counter(degrees)
    degree_values = np.array(sorted(degree_counts.keys()))
    degree_frequencies = np.array([degree_counts[d] for d in degree_values])

    cdf = np.cumsum(degree_frequencies) / len(degrees)
    survival_function = 1 - cdf
    
    survival_function = np.clip(survival_function, a_min=1e-10, a_max=None)
    degree_values = np.clip(degree_values, a_min=1e-10, a_max=None)
    
    log_degree_values = np.log(degree_values)
    log_survival_function = np.log(survival_function)

    slope, intercept, r_value, p_value, std_err = linregress(log_degree_values, log_survival_function)
    return slope * (-1)


def calculate_MLE_estimator(G):
    degrees = [degree for node, degree in G.degree()]
    min_degree = min(degrees)
    
    sum_log = np.sum(np.log(np.array(degrees) / min_degree))
    alpha_mle = 1 + len(degrees) / sum_log

    print(f"Estimated alpha (MLE): {alpha_mle:.2f}")