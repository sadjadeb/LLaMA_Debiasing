import random
from mdp import State


def get_agent_ranking_list(agent, qid, df):
    """
    Run agent to rank a whole (single) query and get list
    agent: DQN agent
    qid: string query id4
    """
    filtered_df = df.loc[df["qid"] == int(qid)].reset_index()
    remaining = list(filtered_df["doc_id"])
    random.shuffle(remaining)
    state = State(0, qid, remaining)
    ranking = []
    t = 0
    while len(remaining) > 0:
        next_action = agent.get_action(state, df)
        t += 1
        remaining.remove(next_action)
        state = State(t, qid, remaining)
        ranking.append(next_action)
    return ranking


def write_trec_results(agent, dataset, feature_name, output_file_path: str):
    with open(output_file_path, 'w') as file:
        for qid in set(dataset["qid"]):
            agent_ranking = get_agent_ranking_list(agent, qid, dataset)
            for rank, doc_id in enumerate(agent_ranking, start=1):
                relevance_score = dataset[(dataset["qid"] == qid) & (dataset["doc_id"] == doc_id)][feature_name].values[0]
                file.write(f"{qid} QO {doc_id} {rank} {relevance_score} ModelName\n")
