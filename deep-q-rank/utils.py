import pandas as pd
from typing import List
import numpy as np
import matplotlib.pyplot as plt


def load_dataset(read_file: str, top_docs_count: int) -> pd.DataFrame:
    """
    Load and preprocess dataset.
    
    Args:
    - read_file (str): Path to the input file.
    - top_docs_count (int): Number of top documents to consider for each query.
    - save_file (str): Path to save the processed file for possible future use.
    
    Returns:
    - pd.DataFrame: Processed dataset.
    """

    dic = {"qid": [], "doc_id": [], "relevance": [], "bias": []}

    for i in range(1, 769):
        dic[i] = []

    with open(read_file, 'r', encoding='utf8') as f:
        qid_set = set()
        for line in f:
            qrel = line.strip().split(" ")
            qid = qrel[0]
            if qid not in qid_set:
                qid_set.add(qid)
                row_counter = 1

            if row_counter > top_docs_count:
                continue
            else:
                doc_id, relevance, bias = qrel[2], qrel[4], float(qrel[6])
                dic["qid"].append(int(qid))
                dic["doc_id"].append(doc_id)
                dic["relevance"].append(relevance)
                dic["bias"].append(bias)
                for i in range(1, 47):
                    dic[i].append(0.0)
                row_counter += 1

    df = pd.DataFrame(data=dic).sort_values(["qid", "relevance"], ascending=False)
    print("Loaded data from run file")

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
