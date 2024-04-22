import pandas as pd
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

# %%
df = pd.read_csv('./data/selected/formatted_documents.csv', sep=',', usecols=['full_text_adj'])
doc_text = df.iloc[0]['full_text_adj']
# %%

model_name = "robinsmits/open_llama_7b_alpaca_clean_dutch_qlora"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, add_eos_token=True)
# config = PeftConfig.from_pretrained(model_name)
print(f"Loaded tokenizer: {model_name}")
print(f"Loading model: {model_name}")

model = AutoPeftModelForCausalLM.from_pretrained(model_name).to('cuda')
# model = PeftModel.from_pretrained(model, model_name)
# %%
# model.to('cuda')
# %%
instruction_eng = "Extract the content from the document below and remove headers and footers"
prompt_eng = f'### Instruction:\n{instruction_eng}\n """{doc_text}"""\n\n### Answer:\n'
instruction = "Pak de inhoud van het onderstaande document uit en verwijder kop- en voetteksten"
prompt = f'### Instructie:\n{instruction}\n """{doc_text}"""\n\n### Antwoord:\n'
print(prompt)
# %%
# t5_name = "yhavinga/t5-v1.1-large-dutch-cnn-test"
# model = AutoModelForSeq2SeqLM.from_pretrained(t5_name)
# %%

# print("Prompt:", prompt)
# %%
inputs = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
# %%
sample = model.generate(input_ids=inputs, num_beams=2, early_stopping=False,
                        eos_token_id=tokenizer.eos_token_id)
# %%
output = tokenizer.decode(sample[0], skip_special_tokens=True)
# %%

print(output.replace('. ', '.\n'))

# print(output.split(prompt)[1])
# %%
