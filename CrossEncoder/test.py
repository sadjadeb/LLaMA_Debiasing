from sentence_transformers import CrossEncoder
import os
import sys
from tqdm import tqdm
import ir_measures
from ir_measures import *

LOCAL = True if sys.platform == 'win32' else False
# First, we define the transformer model we want to fine-tune
model_name = 'bert-base-uncased'
model_save_path = f'output/cross-encoder_{model_name.split("/")[-1]}'
run_output_path = model_save_path + '/Run.txt'
device = 'cpu' if LOCAL else 'cuda:1'

model = CrossEncoder(model_save_path, device=device, num_labels=1)
print(f'{model_save_path} model loaded.')

# Now we read the MS Marco dataset
data_folder = '/home/sajadeb/msmarco'

# Read the corpus files, that contain all the passages. Store them in the corpus dict
print('Loading collection...')
collection = {}
collection_filepath = os.path.join(data_folder, 'collection.tsv')
with open(collection_filepath, 'r', encoding='utf8') as f:
    for line in f:
        pid, passage = line.strip().split("\t")
        collection[pid] = passage.strip()

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
qrels_filepath = os.path.join(data_folder, 'runbm25anserini.dev')
with open(qrels_filepath, 'r', encoding='utf8') as f:
    for line in f:
        qrel = line.strip().split(" ")
        qid = qrel[0]
        pid = qrel[2]
        if qid in qrels:
            qrels[qid].append(pid)
        else:
            qrels[qid] = [pid]

# Search in a loop for the individual queries
ranks = {}
cnt = 0
for qid, passages in tqdm(qrels.items()):
    query = queries[qid]

    # Concatenate the query and all passages and predict the scores for the pairs [query, passage]
    model_inputs = [[query, collection[pid]] for pid in passages]
    scores = model.predict(model_inputs)

    # Sort the scores in decreasing order
    results = [{'pid': pid, 'score': score} for pid, score in zip(passages, scores)]
    results = sorted(results, key=lambda x: x['score'], reverse=True)
    ranks[qid] = results

print('Writing the result to file...')
with open(run_output_path, 'w', encoding='utf-8') as out:
    for qid, results in ranks.items():
        for rank, hit in enumerate(results):
            out.write(f'{qid} Q0 {hit["pid"]} {rank + 1} {hit["score"]} CrossEncoder\n')

print('Evaluation...')
qrels = ir_measures.read_trec_qrels(os.path.join(data_folder, 'qrels.dev.small.tsv'))
run = ir_measures.read_trec_run(run_output_path)
print(ir_measures.calc_aggregate([nDCG@10, P@10, AP@10, RR@10, R@10], qrels, run))
