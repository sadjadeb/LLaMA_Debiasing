import torch
from utils import load_dataset, write_trec_results
from dqn import DQN, DQNAgent
import os
import ir_measures
from ir_measures import *
from calculate_bias import calculate_run_bias, calculate_cumulative_bias

top_docs_count = 10
ndcg_k_list = [10]

test_set_path = "/home/sajadeb/msmarco/run.neutral_queries.trec_with_biases"
bert_model_path = "/home/sajadeb/LLaMA_Debiasing/BiEncoder/output/bi-encoder_margin-mse_bert-base-uncased"
biased_run = "/home/sajadeb/LLaMA_Debiasing/BiEncoder/output/bi-encoder_margin-mse_bert-base-uncased/Run_neutral.txt"
model_save_path = "output/dqn_model/"
pretrained_model_path = os.path.join(model_save_path, "model.pt")
output_file_path = os.path.join(model_save_path, f"Run_nn.txt")


test_set = load_dataset(test_set_path, bert_model_path, top_docs_count, is_training=False)

model = DQN((769,), 1)
model.load_state_dict(torch.load(pretrained_model_path))
agent = DQNAgent((769,), learning_rate=3e-4, buffer=None, dataset=None, pre_trained_model=model)

write_trec_results(agent, test_set, "relevance", output_file_path)

calculate_run_bias.calculate_run_bias(biased_run, output_file_path)
calculate_cumulative_bias.calculate_cumulative_bias()

qrels = ir_measures.read_trec_qrels("/home/sajadeb/msmarco/qrels.dev.tsv")
run = ir_measures.read_trec_run(output_file_path)
ir_measures.calc_aggregate([nDCG@10, P@10, RR@10, R@10, AP@10], qrels, run)
