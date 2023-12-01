import gc
import os
import ir_datasets
import ir_measures
import torch
from ir_measures import *
from sentence_transformers import SentenceTransformer
from tqdm import tqdm, trange
import sys

LOCAL = True if sys.platform == 'win32' else False
model_name = "meta-llama/Llama-2-7b-hf"
model_save_path = f'output/bi-encoder_margin-mse_{model_name.split("/")[-1]}'
run_output_path = model_save_path + '/Run.txt'
device = 'cpu' if LOCAL else 'cuda:1'
batch_size = 128

# Load model
model = SentenceTransformer(model_save_path, device=device)
print(f'{model_save_path} model loaded.')

# Data files
data_folder = '/home/sajadeb/msmarco'

# Read the corpus files, that contain all the passages. Store them in the corpus dict
print('Loading collection...')
corpus = {}
corpus_filepath = os.path.join(data_folder, 'collection.tsv')
with open(corpus_filepath, 'r', encoding='utf8') as f:
    for line in tqdm(f):
        pid, passage = line.strip().split("\t")
        corpus[pid] = passage.strip()

# Read the test queries, store in queries dict
print('Loading queries...')
queries = {}
queries_filepath = os.path.join(data_folder, 'queries.dev.small.tsv')
with open(queries_filepath, 'r', encoding='utf8') as f:
    for line in f:
        qid, query = line.strip().split("\t")
        queries[qid] = query.strip()

print('Loading qrels...')
qrels = {}
pids = set()
qrels_filepath = os.path.join(data_folder, 'runbm25anserini.dev')
with open(qrels_filepath, 'r', encoding='utf8') as f:
    for line in f:
        qrel = line.strip().split(" ")
        qid = qrel[0]
        pid = qrel[2]
        pids.add(pid)
        if qid in qrels:
            qrels[qid].append(pid)
        else:
            qrels[qid] = [pid]


num_batches = (len(pids) + batch_size - 1) // batch_size
pids = list(pids)
embedded_corpus = {}
for i in trange(num_batches):
    batch_pids = pids[i * batch_size: (i + 1) * batch_size]
    batch_texts = [corpus[pid] for pid in batch_pids]
    encoded_batch = torch.tensor(model.encode(batch_texts))

    for idx, pid in enumerate(batch_pids):
        embedded_corpus[pid] = encoded_batch[idx].unsqueeze(0)

embedded_queries = {}
for qid in tqdm(queries):
    query_text = queries[qid]
    embedded_queries[qid] = torch.tensor(model.encode(query_text)).unsqueeze(0)

del corpus
del queries
gc.collect()

# Search in a loop for the individual queries
ranks = {}
for qid, passages in tqdm(qrels.items()):
    query_sentence_embedding = embedded_queries[qid]

    scores = [float(torch.cosine_similarity(query_sentence_embedding, embedded_corpus[pid])) for pid in passages]

    # Sort the scores in decreasing order
    results = [{'pid': pid, 'score': score} for pid, score in zip(passages, scores)]
    results = sorted(results, key=lambda x: x['score'], reverse=True)

    ranks[qid] = results

print('Writing the result to file...')
with open(run_output_path, 'w', encoding='utf-8') as out:
    for qid, results in ranks.items():
        for rank, hit in enumerate(results):
            out.write(f'{qid} Q0 {hit["pid"]} {rank + 1} {hit["score"]} BiEncoder\n')

print('Evaluation...')
qrels = ir_datasets.load('msmarco-passage/dev/small').qrels_iter()
run = ir_measures.read_trec_run(run_output_path)
print(ir_measures.calc_aggregate([nDCG @ 10, P @ 10, AP @ 10, RR @ 10, R @ 10], qrels, run))
