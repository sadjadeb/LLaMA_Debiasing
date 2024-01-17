import torch
from utils import load_dataset
from eval_utils import eval_agent_final, write_trec_results
from dqn import DQN, DQNAgent
from tqdm import tqdm
import os

data_folder = "/home/sajadeb/msmarco"
bert_model_path = "/home/sajadeb/LLaMA_Debiasing/BiEncoder/output/bi-encoder_margin-mse_bert-base-uncased"
model_save_path = "output/dqn_model/"
output_file_path = os.path.join(model_save_path, "Run.txt")
test_set_path = os.path.join(data_folder, "runbm25anserini.dev")
pretrained_model_path = os.path.join(model_save_path, "model.pt")

seed = 0
epochs = 10
window = 1
top_docs_count = 10
ndcg_k_list = 10
fold_list = [1, 10]


def eval_model():
    test_set = load_dataset(test_set_path, bert_model_path, top_docs_count)

    model = DQN((768,), 1)
    model.load_state_dict(torch.load(pretrained_model_path))
    agent = DQNAgent((768,), learning_rate=3e-4, buffer=None, dataset=None, pre_trained_model=model)

    for fold in tqdm(fold_list):
        ndcg_list = eval_agent_final(agent, ndcg_k_list, test_set)
        write_trec_results(agent, test_set, "relevance", output_file_path)
        print(f"Fold {fold} NDCG Values: {ndcg_k_list}")
        print(str(ndcg_list))


if __name__ == "__main__":
    eval_model()
