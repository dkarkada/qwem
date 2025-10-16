import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax
from flax import nnx
from numba import njit

import threading
import queue
from concurrent.futures import ThreadPoolExecutor

import os

from ExptTrace import ExptTrace
from FileManager import FileManager
from misc import rcsetup
rcsetup()

import utils


def get_loss_fn(hypers, min_loss):
    
    def get_sims(model, batch):
        targets, pos_probes, pos_weights, neg_probes, neg_weights = batch
        pos_sim = jnp.einsum('ij,ij->i', model(targets), model(pos_probes))
        neg_sim = jnp.einsum('ij,ij->i', model(targets), model(neg_probes))
        return pos_sim, pos_weights, neg_sim, neg_weights

    def QWEM(model, batch):
        pos_sim, pos_weights, neg_sim, neg_weights = get_sims(model, batch)
        pos_term = (-pos_sim + (1/4)*pos_sim**2) * pos_weights
        neg_term = (neg_sim + (1/4)*neg_sim**2) * neg_weights
        loss = 1 - (pos_term + neg_term).mean()/min_loss
        return loss
    
    def SGNS(model, batch):
        pos_sim, pos_weights, neg_sim, neg_weights = get_sims(model, batch)
        pos_term = jnp.log(1 + jnp.exp(-pos_sim)) * pos_weights
        neg_term = jnp.log(1 + jnp.exp( neg_sim)) * neg_weights
        const = jnp.log(2)*(pos_weights + neg_weights)
        loss = 1 - (pos_term + neg_term - const).mean()/min_loss
        return loss
    
    if hypers.loss == "qwem":
        return QWEM
    elif hypers.loss == "sgns":
        return SGNS
    else:
        raise ValueError(f"{hypers.loss} is not a supported loss function")


def get_batch_generator(hypers, data_fm, q_pos, q_neg):
    dtype =  np.uint16
    context_len = hypers.context_len
    vocab_sz = hypers.vocab_sz
    batch_sz = hypers.batch_sz
    chunk_narticles = hypers.chunk_narticles
    
    article_idxs = data_fm.load("article_arr_idxs.npy")
    corpus_fn = data_fm.get_filename("corpus.bin")
    chunk = None
    cur_article_idx = 0
    min_article_len = 500
    chunk_nbatches = (0.5 * chunk_narticles * min_article_len * context_len) // batch_sz
    
    def load_chunk(n_threads=8):
        nonlocal chunk, cur_article_idx
        corpus = np.memmap(corpus_fn, dtype=dtype, mode='r')
        
        n_articles = len(article_idxs) - 1
        idxs = np.arange(0, chunk_narticles)
        idxs = (idxs + cur_article_idx*chunk_narticles) % n_articles
        cur_article_idx += 1

        def load_one(idx):
            start, stop = article_idxs[idx], article_idxs[idx+1] - 1
            article = np.array(corpus[start:stop], dtype=dtype)
            return article[article < vocab_sz]

        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            chunk = list(executor.map(load_one, idxs))
        
        del corpus

    @njit
    def pairwise_article(article, idx, targets, pos_probes, pos_weights):
        istep = 20
        jstep = 3
        istart = np.random.randint(istep)
        jstart = 1 + np.random.randint(jstep)
        for i in range(istart, len(article) - (context_len+1), istep):
            w = article[i]
            for j in range(jstart, context_len+1, jstep):
                v = article[i+j]
                targets[idx] = w
                pos_probes[idx] = v
                pos_weights[idx] = context_len+1-j
                idx += 1
                if idx >= len(targets):
                    return idx
        return idx
    
    @njit
    def negative_samples(article, idx, neg_probes):
        istep = 5
        istart = np.random.randint(istep)
        for i in range(istart, len(article), istep):
            w = article[i]
            neg_probes[idx] = w
            idx += 1
            if idx >= len(neg_probes):
                return idx
        return idx
    
    def get_batch():
        targets = np.empty(batch_sz, dtype=dtype)
        pos_probes = np.empty(batch_sz, dtype=dtype)
        pos_weights = np.empty(batch_sz)
        fill_idx = 0
        while fill_idx < batch_sz:
            article = chunk[np.random.randint(chunk_narticles)]
            fill_idx = pairwise_article(article, fill_idx, targets,
                                        pos_probes, pos_weights)
        pos_weights /= (context_len+1)/2
        pos_weights *= q_pos(targets, pos_probes)
        
        neg_probes = np.empty(batch_sz, dtype=dtype)
        fill_idx = 0
        while fill_idx < batch_sz:
            article = chunk[np.random.randint(chunk_narticles)]
            fill_idx = negative_samples(article, fill_idx, neg_probes)
        neg_probes = np.random.permutation(neg_probes)
        neg_weights = hypers.ns_weight * q_neg(targets, neg_probes)
        
        batch = [targets, pos_probes, pos_weights, neg_probes, neg_weights]
        batch = [jax.device_put(x) for x in batch]
        return batch
    
    loaderq = queue.Queue(maxsize=8)    
    def loader():   
        loaderstep = 0
        while True:
            reload = cur_article_idx == 0 or hypers.cycle_chunks
            if reload and loaderstep % chunk_nbatches == 0:
                print("Loading new chunk... ", end="", flush=True)
                load_chunk()
                print("done.", flush=True)
            batch = get_batch()
            loaderq.put(batch)
            loaderstep += 1
    threading.Thread(target=loader, daemon=True).start()
    
    while True:
        yield loaderq.get()


def train_embeddings(hypers):
    H = hypers
    fm = FileManager(H.expt_dir)
    H.save(fm.get_filename("hypers.json"))
    if H.checkpt_intervals:
        savetimes = np.concatenate([
            np.linspace(start, end, num, endpoint=(i==len(H.checkpt_intervals)-1))
            for i, (start, end, num) in enumerate(H.checkpt_intervals)
        ]).astype(int)
        modeldir = f"{H.expt_dir}/models/"
        if not os.path.exists(modeldir):
            os.makedirs(modeldir)

    data_dir = os.path.join(os.getenv("DATASETPATH"), "qwem")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    data_fm = FileManager(data_dir)
    
    analogy_dict = data_fm.load("analogies.pickle")
    if analogy_dict is None:
        raise FileNotFoundError("Analogy file not found.")

    data_fm.set_filepath(H.dataset)
    word_counts = data_fm.load("word_counts.pickle")
    if len(word_counts) < H.vocab_sz:
        raise ValueError(f'Vocab sz {H.vocab_sz} too large. Max = {len(word_counts)}')
    word_counts = word_counts[:H.vocab_sz]
    fm.save(word_counts, "word_counts.pickle")
    vocab = utils.Vocabulary(word_counts)
    unigram = vocab.counts / vocab.counts.sum()
    
    if H.reweight == "unigram":
        distr = (unigram / unigram.mean())**(-1)
        q_pos = lambda i,j: distr[i]*distr[j]
        q_neg = q_pos
        min_loss = -0.5
    elif H.reweight == "omniscient":
        print(f"Loading cooccurrence distribution... ", end="", flush=True)
        corpus_stats = data_fm.load("corpus_stats.pickle")
        if corpus_stats is None:
            raise FileNotFoundError("Corpus stats not found.")
        cL = corpus_stats["context_len"]
        if cL != H.context_len:
            raise ValueError(f"Corpus stats context len {cL} != expt {H.context_len}.")
        Cij, Crwij = corpus_stats["counts"], corpus_stats["counts_reweight"]
        numcounts = Cij[:H.vocab_sz, :H.vocab_sz].sum()
        Pij = Crwij[:H.vocab_sz, :H.vocab_sz] / (numcounts * (cL + 1)/2)
        del Cij, Crwij, corpus_stats
        print("done.")

        print(f"Computing q_pos and q_neg... ", end="", flush=True)
        PiPj = H.ns_weight*np.outer(unigram, unigram)
        Gij = Pij + PiPj
        
        if H.loss == "qwem":
            Mstar = 2*(Pij - PiPj)/(Pij + PiPj)
        elif H.loss == "sgns":
            Mstar = np.log((Pij / PiPj) + np.exp(-5))
        else:
            raise ValueError(f"{H.loss} is not a supported loss function")
        distr = (Gij / Gij.mean())**(-1)
        q_pos = lambda i,j: distr[i, j]
        q_neg = q_pos
        min_loss = -1 * (Mstar**2).mean()
        print("done.")
    elif H.reweight == "sgns":
        accept = np.clip(1e-5/unigram + np.sqrt(1e-5/unigram), 0, 1)
        accept /= accept.mean()
        q_pos = lambda i,j: accept[i]*accept[j]
        neg_distr = unigram**.75 / unigram
        neg_distr /= neg_distr.mean()
        q_neg = lambda i,j: 0.5*(accept[i]*neg_distr[j] + accept[j]*neg_distr[i])
        min_loss = -0.12 # hacky guess lol
    else:
        raise ValueError(f"{H.reweight} is not a supported reweighting scheme")
    
    analogy_dataset = utils.AnalogyDataset(analogy_dict, vocab)
    
    loss_fn = get_loss_fn(H, min_loss)

    batch_gen = get_batch_generator(H, data_fm, q_pos, q_neg)

    initializer = nnx.initializers.normal(stddev=(H.init_sz/np.sqrt(H.embeddim)))
    model = nnx.Embed(H.vocab_sz, H.embeddim, rngs=nnx.Rngs(10),
                      embedding_init=initializer, param_dtype=jnp.float64)
    if H.lr_schedule:
        scheduler = optax.schedules.piecewise_constant_schedule(init_value=H.lr,
                        boundaries_and_scales=H.lr_schedule)
    else:
        scheduler = optax.schedules.constant_schedule(value=H.lr)
    optimizer = nnx.Optimizer(model, optax.sgd(scheduler))

    @nnx.jit
    def train_step(model, optimizer, batch):
        grad_fn = nnx.value_and_grad(loss_fn)
        loss, grads = grad_fn(model, batch)
        optimizer.update(grads)
        return loss
    
    nsteps = 0
    loss_buffer = []
    et_loss, et_acc, et_sv = ExptTrace.multi_init(3, ["nstep"])
    print("Starting training loop.")
    while nsteps <= H.maxsteps:
        batch = next(batch_gen)
        loss = train_step(model, optimizer, batch)
        loss_buffer.append(loss)
        if nsteps % 100 == 0:
            et_loss[nsteps] = np.mean(loss_buffer)
            loss_buffer = []
        if nsteps % 1000 == 0:
            weight = model.embedding.value
            acc = analogy_dataset.eval_accuracy(np.asarray(weight))
            et_acc[nsteps] = acc
            et_sv[nsteps] = np.asarray(jnp.linalg.svdvals(weight[:10_000]))
            results = [x.serialize() for x in [et_loss, et_acc, et_sv]]
            fm.save(results, "results.pickle")
            if nsteps > 0:
                print(f"t={nsteps//1000:03d}k:", end=" ")
                print(f"loss={et_loss[:][-100:].mean():.7f}", end=" ")
                print(f"acc={acc*100:.2f}%")
                utils.make_progress_plot([et_loss, et_acc, et_sv], fm,
                                         title=f"acc {acc*100:.2f}%")
        
        if H.checkpt_intervals and (nsteps in savetimes):
            weight = np.asarray(model.embedding.value)
            np.save(f"{modeldir}/W_{nsteps:08d}.npy", weight)
        nsteps += 1
    if H.checkpt_intervals:
        weight = np.asarray(model.embedding.value)
        np.save(f"{modeldir}/W_final.npy", weight)
