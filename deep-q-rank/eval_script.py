import torch
from utils import load_dataset
from eval_utils import eval_agent_final, write_trec_results
from dqn import DQN, DQNAgent
from tqdm import tqdm
import os

output_path = "output/dqn_model/eval_output.txt"
data_folder = "/home/sajadeb/msmarco"
model_save_path = "output/dqn_model/"
seed = 0
epochs = 10
window = 1
top_docs_count = 10
pretrained_model_path = "output/dqn_model/model.pt"


def eval_model():
    test_set_path = os.path.join(data_folder, "/runbm25anserini_top100_with_biases")
    output_file_path = cfg.eval_output_file_path
    fold_list = cfg.fold_list
    ndcg_k_list = cfg.ndcg_k_list

    test_set = load_dataset(test_set_path, top_docs_count)

    model = DQN((768,), 1)
    model.load_state_dict(torch.load(pretrained_model_path))
    agent = DQNAgent((768,), learning_rate=3e-4, buffer=None, dataset=None, pre_trained_model=model)

    for fold in tqdm(fold_list):
        ndcg_list = eval_agent_final(agent, ndcg_k_list, test_set)
        write_trec_results(agent, test_set, "relevance", output_file_path)
        with open(output_path, "w") as f:
            f.write("Fold {} NDCG Values: {}\n".format(fold, ndcg_k_list))
            f.write(str(ndcg_list))
            f.write("\n")


if __name__ == "__main__":
    eval_model()
