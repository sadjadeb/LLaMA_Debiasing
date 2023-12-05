import gzip
import json
import os
import random
import sys
from peft import get_peft_model, LoraConfig, TaskType
from transformers import LlamaConfig
from sentence_transformers import SentenceTransformer, models, losses, InputExample
from torch.utils.data import Dataset, DataLoader
from tqdm.autonotebook import tqdm

data_folder = '/home/sajadeb/msmarco'
LOCAL = True if sys.platform == 'win32' else False
model_name = "bert-base-uncased"
model_save_path = f'output/bi-encoder_margin-mse_{model_name.split("/")[-1]}'
batch_size = 32
device = 'cpu' if LOCAL else 'cuda:1'
max_seq_length = 512
ce_score_margin = 3
num_epochs = 1
num_negs_per_system = 15
warmup_steps = 1000
use_amp = True

os.makedirs(model_save_path, exist_ok=True)

word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)

if isinstance(word_embedding_model.auto_model.config, LlamaConfig):
    batch_size = 2
    use_amp = False
    peft_config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, inference_mode=False, r=32, lora_alpha=64, lora_dropout=0.1)
    word_embedding_model.auto_model = get_peft_model(word_embedding_model.auto_model, peft_config)
    word_embedding_model.auto_model.print_trainable_parameters()
    word_embedding_model.tokenizer.pad_token = word_embedding_model.tokenizer.eos_token
    word_embedding_model.auto_model.config.pad_token_id = word_embedding_model.tokenizer.pad_token_id

pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)

# Read the corpus files, that contain all the passages. Store them in the corpus dict
corpus = {}
collection_filepath = os.path.join(data_folder, 'collection.tsv')
with open(collection_filepath, 'r', encoding='utf8') as f:
    print('Loading collection...')
    for line in f:
        pid, passage = line.strip().split("\t")
        corpus[pid] = {'text': passage, 'title': ''}

# Read the train queries, store in queries dict
queries = {}
queries_filepath = os.path.join(data_folder, 'queries.train.tsv')
with open(queries_filepath, 'r', encoding='utf8') as fIn:
    print('Loading queries...')
    for line in fIn:
        qid, query = line.strip().split("\t")
        queries[qid] = query

print("Loading MSMARCO hard-negatives...")
msmarco_triplets_filepath = os.path.join(data_folder, "msmarco-hard-negatives.jsonl.gz")
train_queries = {}
with gzip.open(msmarco_triplets_filepath, 'rt', encoding='utf8') as fIn:
    for line in tqdm(fIn, total=502939):
        data = json.loads(line)

        # Get the positive passage ids
        pos_pids = [item['pid'] for item in data['pos']]
        pos_min_ce_score = min([item['ce-score'] for item in data['pos']])
        ce_score_threshold = pos_min_ce_score - ce_score_margin

        # Get the hard negatives
        neg_pids = set()
        for system_negs in data['neg'].values():
            negs_added = 0
            for item in system_negs:
                if item['ce-score'] > ce_score_threshold:
                    continue

                pid = item['pid']
                if pid not in neg_pids:
                    neg_pids.add(pid)
                    negs_added += 1
                    if negs_added >= num_negs_per_system:
                        break

        if len(pos_pids) > 0 and len(neg_pids) > 0:
            train_queries[data['qid']] = {'query': queries[data['qid']], 'pos': pos_pids, 'hard_neg': list(neg_pids)}


class MSMARCODataset(Dataset):
    def __init__(self, queries, corpus):
        self.queries = queries
        self.queries_ids = list(queries.keys())
        self.corpus = corpus

        for qid in self.queries:
            self.queries[qid]['pos'] = list(self.queries[qid]['pos'])
            self.queries[qid]['hard_neg'] = list(self.queries[qid]['hard_neg'])
            random.shuffle(self.queries[qid]['hard_neg'])

    def __getitem__(self, item):
        query = self.queries[self.queries_ids[item]]
        query_text = query['query']

        pos_id = query['pos'].pop(0)  # Pop positive and add at end
        pos_text = self.corpus[pos_id]["text"]
        query['pos'].append(pos_id)

        neg_id = query['hard_neg'].pop(0)  # Pop negative and add at end
        neg_text = self.corpus[neg_id]["text"]
        query['hard_neg'].append(neg_id)

        return InputExample(texts=[query_text, pos_text, neg_text])

    def __len__(self):
        return len(self.queries)


train_dataset = MSMARCODataset(train_queries, corpus=corpus)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

# Training with cosine-product
train_loss = losses.MultipleNegativesRankingLoss(model=model)
# training with dot-product
# train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model, similarity_fct=util.dot_score, scale=1)

model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=num_epochs,
          warmup_steps=warmup_steps,
          optimizer_params={'lr': 2e-5, 'eps': 1e-6, 'correct_bias': False},
          output_path=model_save_path,
          use_amp=use_amp)

print(f"Model saved at {model_save_path}")
