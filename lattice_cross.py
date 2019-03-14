#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import tensorflow as tf
import tensorflow_lattice as tfl
import re
import random
from Bio import SeqIO
import gzip
import pandas as pd
import functools
import shutil
from matplotlib import pyplot
import multiprocessing as mp
import bisect
import time

# In[2]:


numpy_path = "../data/np_train/"
fasta_path = "../data/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta.gz"
model_dir = "./model_cross/"
quantiles_dir = "./quantiles_cross/"


# In[3]:


chromosomes = [f"chr{i}" for i in range(1, 23)] + ["chrX"]


# In[4]:


def window(window_size):
    return range(-(window_size // 2), window_size // 2 + 1)


# In[5]:


num_cell_lines = 51
cell_line_keypoints = 64
num_assays = 35
assay_keypoints = 64


# In[6]:


s = """chr1    248956422
chr2    242193529
chr3    198295559
chr4    190214555
chr5    181538259
chr6    170805979
chr7    159345973
chr8    145138636
chr9    138394717
chr10    133797422
chr11    135086622
chr12    133275309
chr13    114364328
chr14    107043718
chr15    101991189
chr16    90338345
chr17    83257441
chr18    80373285
chr19    58617616
chr20    64444167
chr21    46709983
chr22    50818468
chrX    156040895"""
chrom_length_map = {
    k: int(v) for k, v in (line.split() for line in s.split("\n"))
}


# In[7]:


base_symbols = list("ACGTWSMKRYBDHVNZ")


# In[8]:


references = {}
with gzip.open(fasta_path, "rt") as seq_file:
    for seq in SeqIO.parse(seq_file, "fasta"):
        references[seq.id] = seq.upper()


# In[9]:


missing_val = -100.0

def score_transform(s):
    if np.isnan(s):
        return missing_val
#     return np.log(s + 1e-10)
    # sigmoid transform to squish ends, centered about 0.05
    return np.tanh(s + np.log(0.05))

def find_value(arr, key):
    ind = bisect.bisect_left(arr["f0"], key)
    if ind == len(arr):
        return np.nan
    l, h, v = arr[ind]
    if l <= key < h:
        return v
    if l > key:
        if ind > 0:
            l, h, v = arr[ind - 1]
            if l <= key < h:
                return v
    return np.nan

def get_filename(line, assay, chrom):
    return os.path.abspath(os.path.join(numpy_path, f"{line}_{assay}_{chrom}.npy"))


# In[10]:


def create_feature_columns(window_size):
    chrom = tf.feature_column.categorical_column_with_vocabulary_list("chr", chromosomes)
    assays = tf.feature_column.categorical_column_with_vocabulary_list("assay", [str(i) for i in range(1, 36)])
    lines = tf.feature_column.categorical_column_with_vocabulary_list("line", [str(i) for i in range(1, 52)])
    base_features = [
        tf.feature_column.categorical_column_with_vocabulary_list(f"base_{i}", base_symbols)
        for i in range(-(window_size // 2), window_size // 2 + 1)
    ]
    cell_features = [tf.feature_column.numeric_column(f"C{i}") for i in range(1, num_cell_lines+1)]
    assay_features = [tf.feature_column.numeric_column(f"M{i}") for i in range(1, num_assays+1)]
    return [chrom, assays, lines] + base_features + cell_features + assay_features

def get_random_input(num_chrom_samples, sample_intervals, window_size):
    fnames = random.choices(os.listdir(numpy_path), k=num_chrom_samples)
    lines = []
    chrs = []
    assays = []
    scores = []
    bases = [[] for _ in range(window_size)]
    cell_line_values = [[] for _ in range(num_cell_lines)]
    assay_values = [[] for _ in range(num_assays)]
    
    
    for fname in fnames:
        line, assay, chrom = re.match(r"(\d+)_(\d+)_(chr.+)\.npy", fname).groups()
        arr = np.load(os.path.join(numpy_path, fname), mmap_mode="r")
        sampled = arr[np.random.choice(len(arr), sample_intervals)]
        rel_centers = np.random.uniform(sample_intervals)
        interval_centers = np.rint(sampled["f0"] * rel_centers + (1 - rel_centers) * sampled["f1"]).astype(np.int32)
        clipped = np.clip(interval_centers, window_size // 2, chrom_length_map[chrom] - window_size // 2 - 1)
        chrs.extend([chrom] * sample_intervals)
        assays.extend([assay] * sample_intervals)
        lines.extend([line] * sample_intervals)
        for center, score in zip(clipped, sampled["f2"]):
            scores.append(score_transform(score))
            subseq = references[chrom][center - window_size // 2: center + window_size//2 + 1]
            for lst, char in zip(bases, subseq):
                lst.append(char)
        
        for cell_line in range(1, num_cell_lines+1):
            neighbor_filename = get_filename(cell_line, assay, chrom)
            if cell_line == int(line, 10) or not os.path.exists(neighbor_filename):
                values = [missing_val] * sample_intervals
            else:
                arr = np.load(neighbor_filename)
                values = [score_transform(find_value(arr, center)) for center in clipped]
            cell_line_values[cell_line-1].extend(values)
        
        for assay_id in range(1, num_assays+1):
            neighbor_filename = get_filename(line, assay_id, chrom)
            if assay_id == int(assay, 10) or not os.path.exists(neighbor_filename):
                values = [missing_val] * sample_intervals
            else:
                arr = np.load(neighbor_filename)
                values = [score_transform(find_value(arr, center)) for center in clipped]
            assay_values[assay_id-1].extend(values)
        
    
    cols = {
        "chr": chrs,
        "assay": assays,
        "scores": scores,
        "line": lines
    }
    for i, l in zip(range(-(window_size // 2), window_size // 2 + 1), bases):
        cols[f"base_{i}"] = l
    
    cols.update({f"C{i}":cvs for i, cvs in enumerate(cell_line_values, start=1)})
    cols.update({f"M{i}":avs for i, avs in enumerate(assay_values, start=1)})
    
    return pd.DataFrame(cols)

def parallel_get_random_input(num_chrom_samples, sample_intervals, window_size):
    chrom_par = min(8, num_chrom_samples)
    interval_par = 1
    with mp.Pool(chrom_par) as pool:
        args = [(num_chrom_samples // chrom_par, sample_intervals // interval_par, window_size)]
        args = args * chrom_par * interval_par
        return pd.concat(pool.starmap(get_random_input, args), ignore_index=True)

def get_input_fn(num_chrom_samples, sample_intervals, window_size, threads=16):
    with pd.HDFStore(f"train_65536_{window_size}.hdf") as hdfs:
        df = hdfs.get(random.choice(hdfs.keys()))
    # df = parallel_get_random_input(num_chrom_samples, sample_intervals, window_size)
    return tf.estimator.inputs.pandas_input_fn(
        x=df.drop(axis=1, columns="scores"),
        y=df.scores,
        batch_size=len(df),
        num_threads=threads,
        shuffle=True
    )


# In[11]:


def create_calibrated_etl(window_size, config, lattice_rank=4):
    feature_columns = create_feature_columns(window_size)
    feature_names = [fc.name for fc in feature_columns]
    hparams = tfl.CalibratedEtlHParams(
        feature_names=feature_names,
        learning_rate=0.02,
        calibration_l2_laplacian_reg=1.0e-4,
        lattice_l2_laplacian_reg=1.0e-5,
        lattice_l2_torsion_reg=1.0e-5,
        optimizer=tf.train.AdamOptimizer,
        non_monotonic_num_lattices=128,
        interpolation_type="simplex",
        non_monotonic_lattice_size=4,
        non_monotonic_lattice_rank=2,
        missing_input_value=missing_val
    )
    hparams.set_feature_param("chr", "num_keypoints", 23)
    hparams.set_feature_param("assay", "num_keypoints", 35)
    hparams.set_feature_param("line", "num_keypoints", 51)
    for offset in window(window_size):
        hparams.set_feature_param(f"base_{offset}", "num_keypoints", len(base_symbols))
    
    for i in range(1, num_cell_lines+1):
        hparams.set_feature_param(f"C{i}", "num_keypoints", cell_line_keypoints)
#         hparams.set_feature_param(f"C{i}", "missing_input_value", missing_val)
    
    for i in range(1, num_assays+1):
        hparams.set_feature_param(f"M{i}", "num_keypoints", assay_keypoints)
#         hparams.set_feature_param(f"M{i}", "missing_input_value", missing_val)
    
    return tfl.calibrated_etl_regressor(
        feature_columns=feature_columns,
        model_dir=model_dir, config=config, hparams=hparams,
        quantiles_dir=os.path.join(quantiles_dir, str(window_size))
    )

def create_calibrated_rtl(window_size, config, lattice_rank=4):
    feature_columns = create_feature_columns(window_size)
    feature_names = [fc.name for fc in feature_columns]
    hparams = tfl.CalibratedRtlHParams(
        feature_names=feature_names,
        learning_rate=0.02,
        calibration_l2_laplacian_reg=1.0e-4,
        lattice_l2_laplacian_reg=1.0e-5,
        lattice_l2_torsion_reg=1.0e-5,
        optimizer=tf.train.AdamOptimizer,
        num_lattices=1024,
        interpolation_type="hypercube",
        lattice_size=4,
        lattice_rank=4,
        missing_input_value=missing_val
    )
    hparams.set_feature_param("chr", "num_keypoints", 23)
    hparams.set_feature_param("assay", "num_keypoints", 35)
    hparams.set_feature_param("line", "num_keypoints", 51)
    for offset in window(window_size):
        hparams.set_feature_param(f"base_{offset}", "num_keypoints", len(base_symbols))
    
    for i in range(1, num_cell_lines+1):
        hparams.set_feature_param(f"C{i}", "num_keypoints", cell_line_keypoints)
#         hparams.set_feature_param(f"C{i}", "missing_input_value", missing_val)
    
    for i in range(1, num_assays+1):
        hparams.set_feature_param(f"M{i}", "num_keypoints", assay_keypoints)
#         hparams.set_feature_param(f"M{i}", "missing_input_value", missing_val)
    
    return tfl.calibrated_rtl_regressor(
        feature_columns=feature_columns,
        model_dir=model_dir, config=config, hparams=hparams,
        quantiles_dir=os.path.join(quantiles_dir, str(window_size))
    )


# In[12]:


def train(estimator, iters, window_size, num_chrom_samples, sample_intervals):
    input_fn = functools.partial(get_input_fn, num_chrom_samples, sample_intervals, window_size)
    for iter_cnt in range(iters):
        print("Start:", time.time())
        estimator.train(input_fn=input_fn())
        print("Train iter:", time.time())
        print(f"Finished {iter_cnt}/{iters}.")
        evaluation = estimator.evaluate(input_fn=input_fn())
        print("Eval:", time.time())
        print(f"Current average_loss: {evaluation['average_loss']}")
        


# In[13]:


window_size = 21
num_chrom_samples = 4096
sample_intervals = 16
iterations = 32


# In[14]:


quantiles_path = os.path.join(quantiles_dir, str(window_size))
if os.path.exists(quantiles_path):
    shutil.rmtree(quantiles_path)
tfl.save_quantiles_for_keypoints(
    input_fn = get_input_fn(num_chrom_samples, sample_intervals, window_size),
    save_dir=quantiles_path,
    feature_columns=create_feature_columns(window_size),
    num_steps=None)


# In[15]:


# if os.path.exists(model_dir):
#     print("Removing Model Directory")
#     shutil.rmtree(model_dir)
print("Starting run")
config = tf.estimator.RunConfig().replace(model_dir=model_dir)
model = create_calibrated_rtl(window_size, config)
train(model, iterations, window_size, num_chrom_samples, sample_intervals)


# In[ ]:




