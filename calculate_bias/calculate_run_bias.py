import numpy as np
import pickle

# Paths to TREC run files
experiments = {'BM25': '/home/sajadeb/msmarco/runbm25anserini.dev',
               'BERT': 'BiEncoder/output/bi-encoder_margin-mse_bert-base-uncased/Run.txt',
               'DQN': 'deep-q-rank/output/dqn_model/Run_10.txt',
               }

docs_bias_paths = {'tf': "documents_bias_tf.pkl"}

at_rank_list = [5, 10]
save_path_base = "deep-q-rank/output/"
queries_gender_annotated_path = "../resources/queries_gender_annotated.csv"

# Loading saved document bias values
docs_bias = {}
for _method in docs_bias_paths:
    print(_method)
    with open(docs_bias_paths[_method], 'rb') as fr:
        docs_bias[_method] = pickle.load(fr)

# Loading gendered queries
qids_filter = []
with open(queries_gender_annotated_path, 'r') as fr:
    for line in fr:
        vals = line.strip().split(',')
        qid = int(vals[0])
        qids_filter.append(qid)

qids_filter = set(qids_filter)
print(len(qids_filter))

# Loading run files
runs_docs_bias = {}
for exp_name in experiments:
    run_path = experiments[exp_name]
    runs_docs_bias[exp_name] = {}

    for _method in docs_bias_paths:
        runs_docs_bias[exp_name][_method] = {}

    with open(run_path) as fr:
        qid_cur = 0
        for i, line in enumerate(fr):
            vals = line.strip().split(' ')
            if len(vals) == 6:
                qid = int(vals[0])
                doc_id = int(vals[2])

                if qid not in qids_filter:
                    continue

                print('reach')
                if qid != qid_cur:
                    for _method in docs_bias_paths:
                        runs_docs_bias[exp_name][_method][qid] = []
                    qid_cur = qid
                for _method in docs_bias_paths:
                    print(docs_bias[_method][doc_id])
                    runs_docs_bias[exp_name][_method][qid].append(docs_bias[_method][doc_id])

    for _method in docs_bias_paths:
        print(f"Number of effective queries in {exp_name} using {_method} : {len(runs_docs_bias[exp_name][_method].keys())}")

print(runs_docs_bias.keys())


def calc_RaB_q(bias_list, at_rank):
    bias_val = np.mean([x[0] for x in bias_list[:at_rank]])
    bias_feml_val = np.mean([x[1] for x in bias_list[:at_rank]])
    bias_male_val = np.mean([x[2] for x in bias_list[:at_rank]])

    return bias_val, bias_feml_val, bias_male_val


def calc_ARaB_q(bias_list, at_rank):
    _vals = []
    _feml_vals = []
    _male_vals = []
    for t in range(at_rank):
        if len(bias_list) >= t + 1:
            _val_RaB, _feml_val_RaB, _male_val_RaB = calc_RaB_q(bias_list, t + 1)
            _vals.append(_val_RaB)
            _feml_vals.append(_feml_val_RaB)
            _male_vals.append(_male_val_RaB)

    bias_val = np.mean(_vals)
    bias_feml_val = np.mean(_feml_vals)
    bias_male_val = np.mean(_male_vals)

    return bias_val, bias_feml_val, bias_male_val


print('Calculating ranking bias ...')
query_bias_ARaB = {}
for exp_name in experiments:
    query_bias_ARaB[exp_name] = {}

    for _method in docs_bias_paths:
        print(exp_name, _method)

        query_bias_ARaB[exp_name][_method] = {}

        for at_rank in at_rank_list:
            query_bias_ARaB[exp_name][_method][at_rank] = {}

            for qid in runs_docs_bias[exp_name][_method]:
                query_bias_ARaB[exp_name][_method][at_rank][qid] = calc_ARaB_q(runs_docs_bias[exp_name][_method][qid], at_rank)

print('Saving results ...')
for exp_name in experiments:
    for _method in docs_bias_paths:
        save_path = save_path_base + f"run_bias_{exp_name}_{_method}"
        print(save_path)

        with open(save_path + '_ARaB.pkl', 'wb') as fw:
            pickle.dump(query_bias_ARaB[exp_name][_method], fw)
