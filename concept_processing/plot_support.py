import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    
    
def plot_rank_versus_freq(term_counts, yjitter=0,**kwargs):
    yvals = np.sort(term_counts)[::-1]
    xvals = np.arange(1,term_counts.size+1)
    if yjitter > 0:
        yvals = yvals * np.random.uniform(1-yjitter, 1+yjitter, yvals.size)
    plt.plot(xvals, yvals, **kwargs)
    plt.xlim([1,None])
    plt.ylim([1,None])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Rank")
    plt.ylabel("Count")

