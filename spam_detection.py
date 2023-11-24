import pandas as pd
import torch
from peft import get_peft_model, LoraConfig, TaskType
from transformers import LlamaTokenizer, LlamaForSequenceClassification
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tqdm import tqdm

model_name = "meta-llama/Llama-2-7b-hf"
device_type = 'cuda:1'
device = torch.device(device_type)
batch_size = 4
epochs = 1
data_path = '../collective_opinion_spam_detection/'

reviews = pd.read_csv(data_path + 'reviewContent', sep='\t', header=None)
reviews.columns = ['user_id', 'product_id', 'date', 'review']

reviews_metadata = pd.read_csv(data_path + 'metadata', sep='\t', header=None)
reviews_metadata.columns = ['user_id', 'product_id', 'rating', 'spam', 'date']
reviews_metadata['spam'] = reviews_metadata['spam'].replace(-1, 0)

df = pd.merge(reviews, reviews_metadata, on=['user_id', 'product_id', 'date'])

train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
print(f"Number of training samples: {len(train_df)}")
print(f"Number of testing samples: {len(test_df)}")

tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForSequenceClassification.from_pretrained(model_name)

peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.2)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
model.to(device)


# Tokenize the review text for train set
train_encodings = tokenizer(list(train_df['review']), truncation=True, padding=True, max_length=256)
train_labels = torch.tensor(list(train_df['spam']))

# Tokenize the review text for test set
test_encodings = tokenizer(list(test_df['review']), truncation=True, padding=True, max_length=256)
test_labels = torch.tensor(list(test_df['spam']))

train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_encodings['input_ids']),
                                               torch.tensor(train_encodings['attention_mask']),
                                               train_labels)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = torch.utils.data.TensorDataset(torch.tensor(test_encodings['input_ids']),
                                              torch.tensor(test_encodings['attention_mask']),
                                              test_labels)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Set the optimizer and learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# Train the model
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader):
        inputs = {'input_ids': batch[0].to(device),
                  'attention_mask': batch[1].to(device),
                  'labels': batch[2].to(device)}

        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss}')

# Evaluate on the test set
model.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for batch in tqdm(test_loader):
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2]}

        outputs = model(**inputs)
        logits = outputs.logits
        predicted_labels = torch.argmax(logits, dim=1)

        predictions.extend(predicted_labels.cpu().numpy())
        true_labels.extend(inputs['labels'].cpu().numpy())

# Calculate precision
precision = precision_score(true_labels, predictions)

# Calculate recall
recall = recall_score(true_labels, predictions)

# Calculate F1 score
f1 = f1_score(true_labels, predictions)

# Calculate accuracy score
accuracy = accuracy_score(true_labels, predictions)

# Print the results
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'Accuracy Score: {accuracy:.4f}')
