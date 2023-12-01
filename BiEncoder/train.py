import sys
import json
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.models import Pooling, Transformer
from sentence_transformers.losses import MarginMSELoss
from torch.utils.data import Dataset
import random
import gzip
import os
import tqdm
import pickle
from transformers import logging, LlamaConfig
import gc
from peft import get_peft_model, LoraConfig, TaskType

logging.set_verbosity_error()

# The  model we want to fine-tune
LOCAL = True if sys.platform == 'win32' else False
model_name = "meta-llama/Llama-2-7b-hf"
model_save_path = f'output/bi-encoder_margin-mse_{model_name.split("/")[-1]}'
batch_size = 32
device = 'cpu' if LOCAL else 'cuda:1'
max_passages = 2e1
max_seq_length = 512
num_negs_per_system = 15
num_epochs = 1
warmup_steps = 1000
use_amp = True

os.makedirs(model_save_path, exist_ok=True)

word_embedding_model = Transformer(model_name, max_seq_length=max_seq_length)

if isinstance(word_embedding_model.auto_model.config, LlamaConfig):
    batch_size = 3
    use_amp = False
    peft_config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, inference_mode=False, r=32, lora_alpha=64, lora_dropout=0.1)
    word_embedding_model.auto_model = get_peft_model(word_embedding_model.auto_model, peft_config)
    word_embedding_model.auto_model.print_trainable_parameters()
    word_embedding_model.tokenizer.pad_token = word_embedding_model.tokenizer.eos_token
    word_embedding_model.auto_model.config.pad_token_id = word_embedding_model.tokenizer.pad_token_id

pooling_model = Pooling(word_embedding_model.get_word_embedding_dimension(), "mean")
model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)

# Now we read the MS Marco dataset
data_folder = '/home/sajadeb/msmarco'

# Read the corpus files, that contain all the passages. Store them in the corpus dict
corpus = {}
collection_filepath = os.path.join(data_folder, 'collection.tsv')
with open(collection_filepath, 'r', encoding='utf8') as f:
    print('Loading collection...')
    for line in f:
        pid, passage = line.strip().split("\t")
        corpus[int(pid)] = passage

# Read the train queries, store in queries dict
queries = {}
queries_filepath = os.path.join(data_folder, 'queries.train.tsv')
with open(queries_filepath, 'r', encoding='utf8') as fIn:
    print('Loading queries...')
    for line in fIn:
        qid, query = line.strip().split("\t")
        queries[int(qid)] = query

# Load a dict (qid, pid) -> ce_score that maps query-ids (qid) and paragraph-ids (pid)
# to the CrossEncoder score computed by the cross-encoder/ms-marco-MiniLM-L-6-v2 model
ce_scores_file = os.path.join(data_folder, 'cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz')
print("Load CrossEncoder scores dict")
with gzip.open(ce_scores_file, 'rb') as fIn:
    ce_scores = pickle.load(fIn)

# As training data we use hard-negatives that have been mined using various systems
hard_negatives_filepath = os.path.join(data_folder, 'msmarco-hard-negatives.jsonl.gz')
print("Read hard negatives train file")
train_queries = {}
negs_to_use = ['msmarco-distilbert-base-v3']
with gzip.open(hard_negatives_filepath, 'rt') as fIn:
    for line in tqdm.tqdm(fIn):
        if 0 < max_passages <= len(train_queries):
            break
        data = eval(line) if LOCAL else json.loads(line)

        # Get the positive passage ids
        pos_pids = data['pos']

        # Get the hard negatives
        neg_pids = set()
        if negs_to_use is None:
            negs_to_use = list(data['neg'].keys())
            print("Using negatives from the following systems:", negs_to_use)

        for system_name in negs_to_use:
            if system_name not in data['neg']:
                continue

            system_negs = data['neg'][system_name]
            negs_added = 0
            for pid in system_negs:
                if pid not in neg_pids:
                    neg_pids.add(pid)
                    negs_added += 1
                    if negs_added >= num_negs_per_system:
                        break

        if len(pos_pids) > 0 and len(neg_pids) > 0:
            train_queries[data['qid']] = {'qid': data['qid'], 'query': queries[data['qid']], 'pos': pos_pids,
                                          'neg': neg_pids}

print("Train queries: {}".format(len(train_queries)))


class MSMARCODataset(Dataset):
    def __init__(self, queries, corpus, ce_scores):
        self.queries = queries
        self.queries_ids = list(queries.keys())
        self.corpus = corpus
        self.ce_scores = ce_scores

        for qid in self.queries:
            self.queries[qid]['pos'] = list(self.queries[qid]['pos'])
            self.queries[qid]['neg'] = list(self.queries[qid]['neg'])
            random.shuffle(self.queries[qid]['neg'])

    def __getitem__(self, item):
        query = self.queries[self.queries_ids[item]]
        query_text = query['query']
        qid = query['qid']

        if len(query['pos']) > 0:
            pos_id = query['pos'].pop(0)  # Pop positive and add at end
            pos_text = self.corpus[pos_id]
            query['pos'].append(pos_id)
        else:  # We only have negatives, use two negs
            pos_id = query['neg'].pop(0)  # Pop negative and add at end
            pos_text = self.corpus[pos_id]
            query['neg'].append(pos_id)

        # Get a negative passage
        neg_id = query['neg'].pop(0)  # Pop negative and add at end
        neg_text = self.corpus[neg_id]
        query['neg'].append(neg_id)

        pos_score = self.ce_scores[qid][pos_id]
        neg_score = self.ce_scores[qid][neg_id]

        return InputExample(texts=[query_text, pos_text, neg_text], label=pos_score - neg_score)

    def __len__(self):
        return len(self.queries)


# For training the SentenceTransformer model, we need a dataset, a dataloader, and a loss used for training.
train_dataset = MSMARCODataset(queries=train_queries, corpus=corpus, ce_scores=ce_scores)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, drop_last=True)
train_loss = MarginMSELoss(model=model)

del corpus
del queries
del train_queries
del train_dataset
del ce_scores
gc.collect()

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=num_epochs,
          warmup_steps=warmup_steps,
          use_amp=use_amp,
          optimizer_params={'lr': 2e-5},
          )

# Train latest model
model.save(model_save_path)
