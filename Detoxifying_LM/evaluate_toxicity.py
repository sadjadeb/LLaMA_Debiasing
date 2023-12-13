import evaluate
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

toxicity = evaluate.load("ybelkada/toxicity", "DaNLP/da-electra-hatespeech-detection", module_type="measurement")
ds = load_dataset("OxAISH-AL-LLM/wiki_toxic", split="test")

model_name = "sshleifer/tiny-gpt2"
batch_size = 64
num_samples = 100
context_length = 2000
max_new_tokens = 30
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# consider only toxic prompts
ds = ds.filter(lambda x: x["label"] == 1)

toxicities = {}


model = AutoModelForCausalLM.from_pretrained(model_name, device_map={"": device}, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

input_texts = []
for i, example in enumerate(ds):
    input_text = example["comment_text"][:2000]
    input_texts.append(input_text)

    if i > num_samples:
        break

    if (i + 1) % batch_size == 0:
        inputs = tokenizer(input_texts, return_tensors="pt", padding=True).to(device)
        inputs.input_ids = inputs.input_ids[:context_length]
        inputs.attention_mask = inputs.attention_mask[:context_length]
        outputs = model.generate(**inputs, do_sample=True, max_new_tokens=max_new_tokens, use_cache=True)
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        generated_texts = [generated_text.replace(input_texts[i], "") for i, generated_text in enumerate(generated_texts)]
        toxicity_score = toxicity.compute(predictions=generated_texts)
        input_texts = []

        if model_name not in toxicities:
            toxicities[model_name] = []
        toxicities[model_name].extend(toxicity_score["toxicity"])

# last batch
inputs = tokenizer(input_texts, return_tensors="pt", padding=True).to(device)
outputs = model.generate(**inputs, do_sample=True, max_new_tokens=30)
generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
generated_texts = [generated_text.replace(input_texts[i], "") for i, generated_text in enumerate(generated_texts)]
toxicity_score = toxicity.compute(predictions=generated_texts)
toxicities[model_name].extend(toxicity_score["toxicity"])

# compute mean & std using np
mean = np.mean(toxicities[model_name])
std = np.std(toxicities[model_name])

# print
print(f"Model: {model_name} - Mean: {mean} - Std: {std}")

model = None
torch.cuda.empty_cache()
