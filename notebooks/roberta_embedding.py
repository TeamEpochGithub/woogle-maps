import pickle

import torch
from torch.utils.data import Dataset
from transformers import RobertaForMaskedLM
from transformers import RobertaTokenizer, Trainer, TrainingArguments

# Load the documents with the body text
path_documents = "./datasets/test_documents.pkl"

with open(path_documents, 'rb') as file:
    documents = pickle.load(file)

X_train = documents[0:1000]
X_val = documents[1000:1200]


# create a dataset class for the documents

class CustomDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = item['input_ids'].clone()
        return item

    def __len__(self):
        return len(self.encodings.input_ids)


# fine-tune the pre-train model robert
tokenizer = RobertaTokenizer.from_pretrained("pdelobelle/robbert-v2-dutch-base")
model = RobertaForMaskedLM.from_pretrained("pdelobelle/robbert-v2-dutch-base")

train_data = CustomDataset(tokenizer(X_train, padding=True, truncation=True, return_tensors="pt"))
eval_data = CustomDataset(tokenizer(X_val, padding=True, truncation=True, return_tensors="pt"))

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',  # output directory for model and checkpoints
    num_train_epochs=8,  # total number of training epochs
    per_device_train_batch_size=32,  # batch size per device during training
    logging_dir='./logs',  # directory for storing logs
    logging_steps=10,  # log & save weights each logging_steps
    evaluation_strategy="steps",  # evaluate each `logging_steps`
)

trainer = Trainer(model=model, args=training_args, train_dataset=train_data, eval_dataset=eval_data)

trainer.train()

model.save_pretrained("fine_tuned_roberta_dutch")
tokenizer.save_pretrained("fine_tuned_roberta_dutch_tokenizer")
