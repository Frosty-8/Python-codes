from transformers import BertTokenizer, BertModel #type: ignore
from rich import print as rprint

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

rprint("BERT Output Shape: ", outputs.last_hidden_state.shape)
rprint("First token embedding: ", outputs.last_hidden_state[0][0][:5])