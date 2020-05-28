import matplotlib as mpl
import matplotlib.pyplot as plt
#%matplotlib inline
def plot_trajectories(value, xs, name = None, title = None):
    
    norm = mpl.colors.Normalize(vmin=min(value), vmax=max(value))
    cmap = mpl.cm.ScalarMappable(norm = norm, cmap=mpl.cm.plasma)
    cmap.set_array([])

    for key, trajectory in zip(value, xs):
        plt.scatter(trajectory[:, 0], trajectory[:, 1], 
                        marker = '',
                        zorder=2, 
                        s=50,
                        linewidths=0.2,
                        alpha=.8, 
                        cmap = cmap )
        plt.plot(trajectory[:, 0], trajectory[:, 1], c=cmap.to_rgba(key))
    plt.colorbar(cmap).set_label(name, labelpad=2, size=15 )
    plt.title(title)