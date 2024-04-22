# Load model directly
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
import logging

logging.getLogger('extract_lines').setLevel(logging.INFO)

model_name = "yhavinga/t5-v1.1-large-dutch-cnn-test"
# model_name =
tokenizer = AutoTokenizer.from_pretrained(model_name)
# %%
print(f"Loading model: {model_name}")

# model = AutoModelForSeq2SeqLM.from_pretrained(model_name, num_beams=1)
model = T5ForConditionalGeneration.from_pretrained(model_name, num_beams=2, diversity_penalty=0.5, num_beam_groups=2)
model.to('cuda')
print(f"Model loaded.")
# %%
df = pd.read_csv('./data/selected/formatted_documents.csv', sep=',', usecols=['full_text_adj'])
doc_text = df.iloc[0]['full_text_adj']

instruction_eng = "Extract the content from the document below and remove headers and footers"
prompt_eng = f'### Instruction:\n{instruction_eng}\n """\n{doc_text}\n"""\n\n### Answer:\n'
instruction = "Haal de belangrijkste zinnen uit het volgende document om een extractieve samenvatting te maken"
# instruction = "Pak de inhoud van het onderstaande document uit en verwijder kop- en voetteksten"
instruction_end = "### Extractieve samenvatting:"
prompt = f'### Instructie:\n{instruction}\n """{doc_text}\n"""\n{instruction_end}\n'
print(prompt)

# %%
inputs = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
# %%
sample = model.generate(input_ids=inputs, num_beams=2, early_stopping=False,
                        eos_token_id=tokenizer.eos_token_id)
output = tokenizer.decode(sample[0], skip_special_tokens=True)
# print(output.split(prompt)[1])
print(output.replace('. ', '.\n'))
# %%

from transformers import pipeline

# Initialize the Dutch summarization pipeline
summarizer = pipeline("summarization",  model="t5-base", tokenizer="t5-base")

df = pd.read_csv('data/raw/formatted_documents.csv')
doc_text = df['full_text'].tolist()[100]



# Summarize the text
summary = summarizer(doc_text, max_length=200, min_length=100, do_sample=False)


# Print the summary
print("Original Text:")
print(doc_text)
print("\nSummary:")
print(summary[0]['summary_text'])