import numpy as np
import matplotlib.pyplot as plt

def make_progress_plot(results, fm, title=None):
    plots = [results[0], results[1], results[2], results[2]]

    fig, axes = plt.subplots(2, 2, figsize=(9,6), sharex=True)
    axes = axes.flat
    axes[0].set_ylabel("train loss")
    axes[1].set_ylabel("analogy acc")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("training steps")
    axes[2].set_ylabel("singular vals")
    axes[3].set_xlabel("training steps")
    axes[3].set_ylabel("singular vals")
    
    for ax, etrace in zip(axes, plots):
        ax.set_prop_cycle(None)
        for artist in ax.collections + ax.patches + ax.lines:
            artist.remove()
        tt = etrace.get_axis('nstep')
        ydatas = etrace[:]
        ydatas = np.array(ydatas).T
        ax.set_xlim(0, 1.1*max(1e5, max(tt)))
        if len(ydatas.shape) == 1:
            ydatas = [ydatas]
        for ydata in ydatas:
            ax.plot(tt, ydata)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(fm.get_filename("progress.png"))
    plt.close()