#!/bin/bash

python3 - << EOF
from datasets import load_dataset

# Specify the dataset and cache directory
dataset = load_dataset('wikitext', 'wikitext-103-v1')

# Save the dataset to disk for future use
dataset.save_to_disk('./data')

# Print success message
print("WikiText-103 dataset downloaded and saved successfully to './data'")
EOF
