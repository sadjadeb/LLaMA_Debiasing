import random
import torch
from tqdm import trange
import os

from utils import load_dataset, plot_ma_log10
from dqn import DQNAgent
from mdp import BasicBuffer

data_folder = "/home/sajadeb/msmarco"
model_save_path = "output/dqn_model/"
seed = 0
epochs = 10
window = 1
top_docs_count = 10


def train_model():
    random.seed(seed)

    train_set_path = os.path.join(data_folder, "/runbm25anserini_top100_with_biases")

    train_set = load_dataset(train_set_path, top_docs_count)

    train_buffer = BasicBuffer(30000)
    train_buffer.push_batch(train_set, 3)

    agent = DQNAgent((768,), learning_rate=3e-4, buffer=train_buffer, dataset=train_set)

    y = []
    for _ in trange(epochs):
        y.append(agent.update(1, verbose=True))

    torch.save(agent.model.state_dict(), model_save_path)

    y = [float(x) for x in y]

    print(f"Training Loss: {y}")

    plot_ma_log10(y, window, model_save_path, label="train loss")


if __name__ == "__main__":
    train_model()
