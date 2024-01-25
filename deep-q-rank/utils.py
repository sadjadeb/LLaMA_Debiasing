import pandas as pd
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

data_folder = '/home/sajadeb/msmarco'


def load_dataset(read_file: str, model_save_path: str, top_docs_count: int, is_training: bool) -> pd.DataFrame:
    """
    Load and preprocess dataset.
    
    Args:
    - read_file (str): Path to the input file.
    - top_docs_count (int): Number of top documents to consider for each query.
    - save_file (str): Path to save the processed file for possible future use.
    
    Returns:
    - pd.DataFrame: Processed dataset.
    """

    # Read the corpus files, that contain all the passages. Store them in the corpus dict

    saved_file = os.path.join(data_folder, f'encoded_dataset_{"train" if is_training else "dev_small"}.csv')
    if os.path.exists(saved_file):
        print('Loading saved dataset...')
        df = pd.read_csv(saved_file)
        print('Saved dataset loaded.')
        return df

    print('Loading collection...')
    corpus = {}
    corpus_filepath = os.path.join(data_folder, 'collection.tsv')
    with open(corpus_filepath, 'r', encoding='utf8') as f:
        for line in f:
            pid, passage = line.strip().split("\t")
            corpus[pid] = passage.strip()

    # Read the test queries, store in queries dict
    print('Loading queries...')
    queries = {}
    if is_training:
        queries_filepath = os.path.join(data_folder, 'queries.train.tsv')
    else:
        queries_filepath = os.path.join(data_folder, 'neutral_queries.tsv')
    with open(queries_filepath, 'r', encoding='utf8') as f:
        for line in f:
            qid, query = line.strip().split("\t")
            queries[qid] = query.strip()

    # Load model
    model = SentenceTransformer(model_save_path, device='cuda:3')
    print(f'{model_save_path} model loaded.')

    dic = {"qid": [], "doc_id": [], "relevance": [], "bias": []}

    for i in range(1, 769):
        dic[i] = []

    with open(read_file, 'r', encoding='utf8') as f:
        qid_set = set()
        for line in tqdm(f, total=50289125 if is_training else 1764374):
            data = line.strip().split(" ")
            qid = data[0]

            if qid not in qid_set:
                qid_set.add(qid)
                row_counter = 1

            if row_counter > top_docs_count:
                continue
            else:
                doc_id, relevance, bias = data[2], data[4], float(data[6])
                dic["qid"].append(int(qid))
                dic["doc_id"].append(doc_id)
                dic["relevance"].append(relevance)
                dic["bias"].append(bias)

                vector = model.encode(f'{queries[qid]}[SEP]{corpus[doc_id]}')
                for i in range(1, 769):
                    dic[i].append(vector[i - 1])

                row_counter += 1

    df = pd.DataFrame(data=dic).sort_values(["qid", "relevance"], ascending=False)
    print("Loaded data from run file")

    df.to_csv(saved_file, index=False)

    return df


def get_model_inputs(state, action, dataset) -> np.ndarray:
    """
    Get model inputs for the given state and action.
    
    Args:
    - state: State information.
    - action: Action information.
    - dataset (pd.DataFrame): Dataset for reference.
    
    Returns:
    - np.ndarray: Model inputs.
    """
    return np.array([state.t] + get_features(state.qid, action, dataset))


def get_multiple_model_inputs(state, doc_list, dataset) -> np.ndarray:
    """
    Get multiple model inputs for the given state, list of docs, and dataset.
    
    Args:
    - state: State information.
    - doc_list (List[Union[str, int]]): List of documents.
    - dataset (pd.DataFrame): Dataset for reference.
    
    Returns:
    - np.ndarray: Multiple model inputs.
    """
    return np.insert(get_query_features(state.qid, doc_list, dataset), 0, state.t, axis=1)


def get_features(qid, doc_id, dataset) -> List[float]:
    """
    Get features for the given query id and document id.
    
    Args:
    - qid (Union[str, int]): Query ID.
    - doc_id (Union[str, int]): Document ID.
    - dataset (pd.DataFrame): Dataset for reference.
    
    Returns:
    - List[float]: Features for the given query and document.
    """
    qid, doc_id = int(qid), str(doc_id)
    df = dataset[(dataset["doc_id"].str.contains(doc_id)) & (dataset["qid"] == qid)]
    assert len(df) != 0, "Fix the dataset"

    df_copy = df.copy()
    df_copy.drop(["qid", "doc_id", "relevance"], axis=1, inplace=True)
    df = df_copy
    return df.values.tolist()[0]


def get_query_features(qid, doc_list, dataset) -> np.ndarray:
    """
    Get query features for the given query ID, list of docs, and dataset.
    
    Args:
    - qid (Union[str, int]): Query ID.
    - doc_list (List[Union[str, int]]): List of documents.
    - dataset (pd.DataFrame): Dataset for reference.
    
    Returns:
    - np.ndarray: Query features.
    """
    doc_set = set(doc_list)
    qid = int(qid)
    if len(doc_list) > 0:
        df = dataset[dataset["qid"] == qid]
        df = df[df["doc_id"].isin(doc_set)]
    else:
        df = dataset[dataset["qid"] == qid]
    assert len(df) != 0
    df.drop(["qid", "doc_id", "relevance"], axis=1, inplace=True)
    return df.values


def plot_ma_log10(numbers: List, window: int, plot_name: str, label=""):
    plt.figure(figsize=(10, 6))

    moving_avg = np.convolve(np.log10(numbers), np.ones(window) / window, mode='valid')
    plt.plot(moving_avg)
    plt.grid(True)
    plt.legend()
    plt.title(label)
    plt.savefig(plot_name)
