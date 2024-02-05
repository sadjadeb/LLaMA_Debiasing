import random
import torch
from tqdm import trange
import os

from utils import load_dataset, plot_ma_log10
from dqn import DQNAgent
from mdp import BasicBuffer

data_folder = "/home/sajadeb/msmarco"
model_save_path = "output/dqn_model/"
bert_model_path = "/home/sajadeb/LLaMA_Debiasing/BiEncoder/output/bi-encoder_margin-mse_bert-base-uncased"
train_set_path = os.path.join(data_folder, "runbm25anserini_top100_with_biases")

seed = 42
epochs = 10
window = 1
top_docs_count = 10

random.seed(seed)

train_set = load_dataset(train_set_path, bert_model_path, top_docs_count, is_training=True)

train_buffer = BasicBuffer(30000)
train_buffer.push_batch(train_set, 3)

agent = DQNAgent((769,), learning_rate=3e-4, buffer=train_buffer, dataset=train_set)

y = []
for _ in trange(epochs):
    y.append(agent.update(1))

y = [float(x) for x in y]
print(f"Training Loss: {y}")

plot_ma_log10(y, window, model_save_path + "train_loss", label="train loss")

torch.save(agent.model.state_dict(), os.path.join(model_save_path, "model.pt"))
