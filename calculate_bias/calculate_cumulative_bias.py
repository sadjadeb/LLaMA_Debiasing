import numpy as np
import pickle

experiments = ['BM25', 'BERT', 'DQN']

metrics = ['ARaB']
methods = ['tf']

qry_bias_paths = {}
for metric in metrics:
    qry_bias_paths[metric] = {}
    for exp_name in experiments:
        qry_bias_paths[metric][exp_name] = {}
        for _method in methods:
            qry_bias_paths[metric][exp_name][_method] = f'output/run_bias_{exp_name}_{_method}_{metric}.pkl'

queries_gender_annotated_path = "../resources/queries_gender_annotated.csv"

at_ranklist = [5, 10, 20, 30, 40]

query_bias_per_query = {}

for metric in metrics:
    query_bias_per_query[metric] = {}
    for exp_name in experiments:
        query_bias_per_query[metric][exp_name] = {}
        for _method in methods:
            _path = qry_bias_paths[metric][exp_name][_method]
            print(_path)
            with open(_path, 'rb') as fr:
                query_bias_per_query[metric][exp_name][_method] = pickle.load(fr)

queries_effective = {}
with open(queries_gender_annotated_path, 'r') as fr:
    for line in fr:
        vals = line.strip().split(',')
        qid = int(vals[0])
        qtext = ' '.join(vals[1:-1])
        qgender = vals[-1]
        if qgender == 'n':
            queries_effective[qid] = qtext
len(queries_effective)

eval_results_bias = {}
eval_results_feml = {}
eval_results_male = {}

for metric in metrics:
    eval_results_bias[metric] = {}
    eval_results_feml[metric] = {}
    eval_results_male[metric] = {}
    for exp_name in experiments:
        eval_results_bias[metric][exp_name] = {}
        eval_results_feml[metric][exp_name] = {}
        eval_results_male[metric][exp_name] = {}
        for _method in methods:
            eval_results_bias[metric][exp_name][_method] = {}
            eval_results_feml[metric][exp_name][_method] = {}
            eval_results_male[metric][exp_name][_method] = {}
            for at_rank in at_ranklist:
                _bias_list = []
                _feml_list = []
                _male_list = []

                for qid in queries_effective.keys():
                    if qid in query_bias_per_query[metric][exp_name][_method][at_rank]:
                        _bias_list.append(query_bias_per_query[metric][exp_name][_method][at_rank][qid][0])
                        _feml_list.append(query_bias_per_query[metric][exp_name][_method][at_rank][qid][1])
                        _male_list.append(query_bias_per_query[metric][exp_name][_method][at_rank][qid][2])
                    else:
                        pass
                        # print ('missing', metric, exp_name, _method, at_rank, qid)

                eval_results_bias[metric][exp_name][_method][at_rank] = np.mean(
                    [(_male_x - _feml_x) for _male_x, _feml_x in zip(_male_list, _feml_list)])
                eval_results_feml[metric][exp_name][_method][at_rank] = np.mean(_feml_list)
                eval_results_male[metric][exp_name][_method][at_rank] = np.mean(_male_list)

for metric in metrics:
    print(metric)
    for at_rank in at_ranklist:
        for _method in methods:
            for exp_name in experiments:
                print("%25s\t%2d %5s\t%f\t%f\t%f" % (exp_name,
                                                     at_rank,
                                                     _method,
                                                     eval_results_bias[metric][exp_name][_method][at_rank],
                                                     eval_results_feml[metric][exp_name][_method][at_rank],
                                                     eval_results_male[metric][exp_name][_method][at_rank]))
        print("==========")
