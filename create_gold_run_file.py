from tqdm import tqdm

qrels_file = '/home/sajadeb/msmarco/qrels.train.tsv'
sbert_run_file = '/home/sajadeb/LLaMA_Debiasing/BiEncoder/output/bi-encoder_margin-mse_bert-base-uncased/Run_train_retrieval_top100.txt'
output_file = '/home/sajadeb/LLaMA_Debiasing/BiEncoder/output/bi-encoder_margin-mse_bert-base-uncased/Run_gold_sbert.txt'

print('Reading qrels file...')
qrels = {}
with open(qrels_file, 'r') as f:
    for line in f:
        qid, _, doc_id, rel = line.strip().split()
        if qid not in qrels:
            qrels[qid] = []
        qrels[qid].append(doc_id)

print('Reading sbert run file...')
run = {}
with open(sbert_run_file, 'r') as f:
    for line in f:
        qid, _, doc_id, rank, score, _ = line.strip().split()
        if qid not in run:
            run[qid] = []
        run[qid].append((doc_id, score))

for qid in tqdm(qrels):
    for doc_id in qrels[qid]:
        run[qid].insert(0, (doc_id, 10))

    seen_doc_ids = set()
    new_run = []
    for doc_id, score in run[qid]:
        if doc_id not in seen_doc_ids:
            new_run.append((doc_id, score))
            seen_doc_ids.add(doc_id)

    run[qid] = new_run

print('Writing gold sbert run file...')
with open(output_file, 'w') as f:
    for qid in qrels:
        for rank, (doc_id, score) in enumerate(run[qid]):
            f.write(f'{qid} Q0 {doc_id} {rank + 1} {score} gold_sbert\n')
