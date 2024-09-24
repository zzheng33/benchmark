from transformers import BertForMaskedLM, BertTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
from datasets import load_from_disk
from transformers import AdamW
import random



# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'  # You can change this to 'bert-large-uncased' or other variants
model = BertForMaskedLM.from_pretrained('bert-base-uncased', ignore_mismatched_sizes=True)
tokenizer = BertTokenizer.from_pretrained(model_name)

dataset_path = './data'
dataset = load_from_disk(dataset_path)

# Access the training data
train_dataset = dataset['train']


sample_percentage = 0.01

# Calculate the number of examples to sample (20% of the total training data)
sample_size = int(len(train_dataset) * sample_percentage)

# Set a random seed for reproducibility
random.seed(42)

# Sample 20% of the data
sampled_train_dataset = train_dataset.shuffle(seed=42).select(range(sample_size))

# Access the train, validation, and test splits
val_dataset = dataset['validation']
test_dataset = dataset['test']


# Define a tokenization function
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128, return_tensors='pt')

# Tokenize the dataset
tokenized_datasets = sampled_train_dataset.map(tokenize_function, batched=True, remove_columns=["text"], num_proc=32)
# Convert to PyTorch Dataset for DataLoader
train_dataloader = DataLoader(tokenized_datasets, batch_size=256, shuffle=True, num_workers=32, pin_memory=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set up the optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
model.train()  # Set the model to training mode
for epoch in range(1):  # Loop through each epoch
    for batch in train_dataloader:
     
        

        # Prepare inputs for the model, handling lists of tensors
        inputs = {}
        for key, val in batch.items():
            if isinstance(val, list) and isinstance(val[0], torch.Tensor):
                # Stack the list of tensors into a single tensor
                inputs[key] = torch.stack(val).to(device)
            else:
                # If it's already a tensor, move it to the device
                inputs[key] = val.to(device)

        # Forward pass
        outputs = model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss  # Extract the loss
        loss.backward()  # Backpropagation

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()  # Reset gradients

    # Print epoch completion and loss
    print(f"Epoch {epoch + 1} completed, Loss: {loss.item()}")



