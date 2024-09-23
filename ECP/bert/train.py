from transformers import BertForMaskedLM, BertTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
from datasets import load_from_disk
from transformers import AdamW



# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'  # You can change this to 'bert-large-uncased' or other variants
model = BertForMaskedLM.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

dataset_path = './data'
dataset = load_from_disk(dataset_path)

# Access the train, validation, and test splits
train_dataset = dataset['train']
val_dataset = dataset['validation']
test_dataset = dataset['test']


# Define a tokenization function
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128, return_tensors='pt')

# Tokenize the dataset
tokenized_datasets = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
# Convert to PyTorch Dataset for DataLoader
train_dataloader = DataLoader(tokenized_datasets, batch_size=8, shuffle=True, num_workers=32)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set up the optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
model.train()
for epoch in range(1): 
    for batch in train_dataloader:
        inputs = {key: val.squeeze().to(device) for key, val in batch.items()}

        outputs = model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    print(f"Epoch {epoch + 1} completed, Loss: {loss.item()}")