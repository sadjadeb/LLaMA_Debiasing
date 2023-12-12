import torch
from datasets import load_dataset
from torch.optim import Adam
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, create_reference_model, set_seed

tqdm.pandas()

model_name = "sshleifer/tiny-gpt2"
model_save_path = f"output/{model_name.split('/')[-1]}_detoxified"
dataset_name = "allenai/real-toxicity-prompts"

device = "cuda:3" if torch.cuda.is_available() else "cpu"

# TODO: Replace logging with wandb
config = PPOConfig(
    model_name=model_name,
    learning_rate=1.47e-5,
    log_with=None,
    ppo_epochs=100,
    mini_batch_size=4,
    batch_size=16,
    gradient_accumulation_steps=1,
)

tokenizer = AutoTokenizer.from_pretrained(config.model_name)
# GPT-2 / GPT-J tokenizer has a pad token, but it is not eos_token by default.
tokenizer.pad_token = tokenizer.eos_token

input_size = 35


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def filter_fn(sample):
    toxicity = sample["prompt"]["toxicity"]
    return toxicity is not None and toxicity > 0.3


def tokenize(sample):
    prompt = sample["prompt"]["text"]
    continuation = sample["continuation"]["text"]

    sample["input_ids"] = tokenizer.encode(prompt + continuation)[: input_size]
    sample["query"] = tokenizer.decode(sample["input_ids"])
    return sample


dataset = load_dataset(dataset_name, split="train")
dataset = dataset.filter(filter_fn, batched=False)
dataset = dataset.map(tokenize, batched=False)
dataset.set_format(type="torch")
dataset = dataset.train_test_split(test_size=0.2, shuffle=False)["train"]

# set seed before initializing value head for deterministic eval
set_seed(config.seed)
# TODO: add , torch_dtype=torch.bfloat16 to config
model = AutoModelForCausalLM.from_pretrained(config.model_name).to(device)
model = AutoModelForCausalLMWithValueHead.from_pretrained(model).to(device)

# We create a reference model by sharing 20 layers
ref_model = create_reference_model(model, num_shared_layers=0)

optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)

# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
    optimizer=optimizer,
)
ppo_trainer.current_device = device

# We then build the reward pipeline, we will use the toxicity model to compute the reward.
# We first load the toxicity model and tokenizer.
# TODO: add , torch_dtype=torch.float16 to config
toxicity_model_id = "bert-base-uncased"
toxicity_tokenizer = AutoTokenizer.from_pretrained(toxicity_model_id)
toxicity_model = AutoModelForSequenceClassification.from_pretrained(toxicity_model_id).to(device)

# We then define the arguments to pass to the `generate` function
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": input_size,
}

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]

    # Get response from the policy model
    response_tensors = []
    for query in query_tensors:
        response = ppo_trainer.generate(query, **generation_kwargs)
        response_tensors.append(response.squeeze()[-input_size:])
    batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

    # Compute sentiment score
    texts = batch["response"]
    toxicity_inputs = toxicity_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    logits = toxicity_model(**toxicity_inputs).logits.float()
    toxicity_labels = (logits[:, 0]).tolist()

    rewards = [torch.tensor(output) for output in toxicity_labels]

    # Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

ppo_trainer.save_pretrained(model_save_path)
