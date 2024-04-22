# %%
# Currently, Norambuena's algorithm only embeds the titles. We want to embed the titles and the whole documents in such way
# that we are still able to extract events in a meaningful way. Potentially, we might want to include other features on the embedding too.
import re
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import nltk
import numpy as np
# %%
import pandas as pd
import torch
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import RobertaForMaskedLM, RobertaTokenizer, Trainer, TrainerCallback, TrainingArguments
# %%

nltk.download("stopwords")
nltk.download("punkt")

# %%
# Load the dataset
# For testing purposes only
path = r"data/woo_bodytext.csv.gz"
df = pd.read_csv(path)

df.head()

# %%
# Preprocess the body of text
stop_words = set(stopwords.words("dutch"))


def preprocess_text(text):
    # Additional preprocessing / remove patterns can be considered
    text = re.sub(r"\s+", " ", re.sub(r"[^\w\s]|_", " ", text)).strip()

    tokens = word_tokenize(text)
    filtered_text = [word for word in tokens if word.lower() not in stop_words]
    return " ".join(filtered_text)

    return text


def apply_func_to_series(data_chunk, func, **kwargs):
    return data_chunk.apply(func, **kwargs)


def parallelize_dataframe(series, func, partitions, **kwargs):
    # Split the Series into partitions
    series_split = np.array_split(series, partitions)

    # Create a partial function with the additional kwargs
    func_with_kwargs = partial(apply_func_to_series, func=func, **kwargs)

    # Use ProcessPoolExecutor to parallelize the operation
    with ProcessPoolExecutor() as executor:
        # Result is a list of Series objects
        result_series = pd.concat(executor.map(func_with_kwargs, series_split))

    return result_series


# %%
# Preprocess whole dataset (too much time)
start = time.time()
body_df = df.dropna(subset=["foi_bodyTextOCR"])
body_df["foi_bodyTextOCR"] = parallelize_dataframe(body_df["foi_bodyTextOCR"].astype(str), preprocess_text, 10)
end = time.time()
print(f"Processed {end - start} seconds")

body_df["foi_bodyTextOCR"].to_pickle("data/preprocessed_bodies")


# %%
# Generate custom dataset class


class CustomDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length

        # Tokenize all texts and store the encodings
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = item["input_ids"].clone()
        return item


tokenizer = RobertaTokenizer.from_pretrained("pdelobelle/robbert-v2-dutch-base")
model = RobertaForMaskedLM.from_pretrained("pdelobelle/robbert-v2-dutch-base")
model.to("cuda")


# %%
# Define callback
class ProgressBarCallback(TrainerCallback):
    """A Hugging Face Trainer callback"""

    def __init__(self):
        self.progress_bar = None

    def on_train_begin(self, args, state, control, **kwargs):
        """Initialize the progress bar at the beginning of training."""
        print("Training starts...")
        self.progress_bar = tqdm(total=state.max_steps, desc="Training Progress")

    def on_step_end(self, args, state, control, **kwargs):
        """Update the progress bar at the end of each step."""
        self.progress_bar.update(1)

    def on_train_end(self, args, state, control, **kwargs):
        """Close the progress bar when training ends."""
        self.progress_bar.close()
        print("Training finished!")


# %%
# Generate representative subset
sample_size = 100000
df_sample = df.sample(n=sample_size, random_state=1)

# %%
# Convert df['foi_bodyTextOCR'] to a list of texts for simplicity
sentences = df_sample["foi_bodyTextOCR"].astype(str).apply(preprocess_text).tolist()
dataset = CustomDataset(sentences, tokenizer)

# %%

training_args = TrainingArguments(
    output_dir="./results",  # output directory
    num_train_epochs=1,  # total number of training epochs
    per_device_train_batch_size=4,  # batch size per device during training
    warmup_steps=500,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    logging_dir="./logs",  # directory for storing logs
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    callbacks=[ProgressBarCallback()],
)

trainer.train()

# %%
# Save model to file
model.save_pretrained("models/bert-embedder")
tokenizer.save_pretrained("models/bert-tokenizer")

# %%
# Create embeddings on a single dossier
from transformers import RobertaModel

dossier = pd.read_csv("../data/selected/formatted_documents.csv")
dossier_docs = dossier["full_text"].astype(str).apply(preprocess_text).tolist()

# final_tokenizer = RobertaTokenizer.from_pretrained('../tm/bert-tokenizer')
final_model = RobertaModel.from_pretrained("../tm/bert-embedder")
encoded_input = tokenizer(dossier_docs, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    outputs = model(**encoded_input)

dossier_embeddings = outputs.last_hidden_state
