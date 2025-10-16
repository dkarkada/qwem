from dataclasses import dataclass, field, asdict
import json
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class Hyperparams:
    expt_dir: str
    vocab_sz: int           = 1000
    context_len: int        = 32
    maxsteps: int           = 1_000_000
    lr_schedule: dict       = None
    embeddim: int           = 200
    batch_sz: int           = 20_000
    checkpt_intervals: list = field(default_factory=list)
    lr: float               = 1
    init_sz: float          = 1e-2
    ns_weight: float        = 1.0
    loss: str               = "qwem"
    reweight: str           = "unigram"
    dataset: str            = "enwiki500"
    chunk_narticles: int    = 100_000
    cycle_chunks: bool      = True
    
    def save(self, filepath):
        """Save hyperparameters to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=4)


class Vocabulary:

    def __init__(self, word_counts):
        self.words = np.array([word for word, c in word_counts])
        self.counts = np.array([c for word, c in word_counts])
        self.word2token = {word:tok for tok, word in enumerate(self.words)}
        self.size = len(self.words)

    def get_count(self, word):
        if word not in self.word2token:
            return 0
        return self.counts[self.word2token.get(word)]

    def to_words(self, tokens):
        return " ".join([self.words[tok] for tok in tokens])


class AnalogyDataset:
    
    def __init__(self, analogy_dict, vocab):
        self.pairs = {}
        for category, analogy_list in analogy_dict.items():
            category = category.strip()
            category_pairs = set()
            for analogy in analogy_list:
                a, b, ap, bp = [vocab.word2token.get(word, None) for word in analogy]
                if a and b:
                    category_pairs.add((a, b))
                if ap and bp:
                    category_pairs.add((ap, bp))
            if len(category_pairs) > 1:
                self.pairs[category] = list(category_pairs)
        
        self.analogies = {}
        for category, category_pairs in self.pairs.items():
            category_analogies = []
            for pair1 in category_pairs:
                for pair2 in category_pairs:
                    if pair1 == pair2:
                        continue
                    analogy = [*pair1, *pair2]
                    category_analogies.append(analogy)
            category_analogies = np.array(category_analogies, dtype=int)
            self.analogies[category] = category_analogies
        
        all_analogies = np.concatenate([a for _, a in self.analogies.items()])
        self.analogies["full"] = all_analogies
    
    def get_evals(self, W, category="full", normalize=True):
        normalizer = np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-12)
        probes = W / normalizer
        embeds = probes if normalize else W
        analogies = self.analogies[category]
        w1 = embeds[analogies[:, 0]]
        w2 = embeds[analogies[:, 1]]
        w3 = embeds[analogies[:, 2]]
        pred = probes @ (w3 + w2 - w1).T
        num = len(analogies)
        pred[analogies[:, 0], np.arange(num)] = -np.inf
        pred[analogies[:, 1], np.arange(num)] = -np.inf
        pred[analogies[:, 2], np.arange(num)] = -np.inf
        return (pred.argmax(axis=0) == analogies[:, 3]).astype(int)

    def eval_accuracy(self, W, category="full", normalize=True):
        evals = self.get_evals(W, category, normalize)
        return evals.mean()


def rcsetup():
    plt.rc("figure", dpi=120, facecolor=(1, 1, 1))
    plt.rc("font", family='stixgeneral', size=12)
    plt.rc("axes", facecolor=(1, .99, .95), titlesize=12)
    plt.rc("mathtext", fontset='cm')
    plt.rc("pdf", fonttype=42)  # TrueType
    plt.rc("ps", fonttype=42)


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
