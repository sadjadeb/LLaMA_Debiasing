import numpy as np
import random
from collections import deque


def compute_reward(t, relevance, bias):
    """
    Reward function for MDP
    Bias value is in [-1, 1] -> either do abs or lambda*bias
    Relevance value is in [0, 1]
    """
    if t == 0:
        return 0
    # return relevance / np.log2(t + 1)
    return (float(relevance) * 1000000 / np.log2(t + 1)) + (1 / (abs(bias) + 1))


class State:

    def __init__(self, t, query, remaining):
        self.t = t
        self.qid = query  # useful for sorting buffer
        self.remaining = remaining

    def pop(self):
        return self.remaining.pop()

    def initial(self):
        return self.t == 0

    def terminal(self):
        return len(self.remaining) == 0


class BasicBuffer:

    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def push_batch(self, df, n):
        for i in range(n):
            random_qid = random.choice(list(df["qid"]))
            filtered_df = df.loc[df["qid"] == int(random_qid)].reset_index()
            row_order = [x for x in range(len(filtered_df))]
            zzz = [x[1]["doc_id"] for x in filtered_df.iterrows()]
            random.shuffle(row_order)
            for t, r in enumerate(row_order):
                cur_row = filtered_df.iloc[r]
                old_state = State(t, cur_row["qid"], zzz[:])
                action = cur_row["doc_id"]
                new_state = State(t + 1, cur_row["qid"], zzz[:])
                reward = compute_reward(t + 1, cur_row["relevance"], cur_row["bias"])
                self.push(old_state, action, reward, new_state, t + 1 == len(row_order))
                filtered_df.drop(filtered_df.index[[r]])

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return (state_batch, action_batch, reward_batch,
                next_state_batch, done_batch)

    def __len__(self):
        return len(self.buffer)
