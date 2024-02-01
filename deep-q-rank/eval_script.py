import torch
from utils import load_dataset
from eval_utils import eval_agent_final, write_trec_results
from dqn import DQN, DQNAgent
import os

test_set_path = "/home/sajadeb/msmarco/run.neutral_queries.trec_with_biases"
bert_model_path = "/home/sajadeb/LLaMA_Debiasing/BiEncoder/output/bi-encoder_margin-mse_bert-base-uncased"
model_save_path = "output/dqn_model/"
pretrained_model_path = os.path.join(model_save_path, "model.pt")

top_docs_count = 10
ndcg_k_list = [10]


def eval_model():
    test_set = load_dataset(test_set_path, bert_model_path, top_docs_count, is_training=False)

    model = DQN((769,), 1)
    model.load_state_dict(torch.load(pretrained_model_path))
    agent = DQNAgent((769,), learning_rate=3e-4, buffer=None, dataset=None, pre_trained_model=model)

    ndcg_list = eval_agent_final(agent, ndcg_k_list, test_set)
    output_file_path = os.path.join(model_save_path, f"Run_nn.txt")
    write_trec_results(agent, test_set, "relevance", output_file_path)
    print(f"Fold NDCG Values: {ndcg_list}")


if __name__ == "__main__":
    eval_model()
