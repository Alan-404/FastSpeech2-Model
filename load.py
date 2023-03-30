#%%
from preprocessing.text import TextProcessor
# %%
text_processor = TextProcessor("./tokenizer/tokenizer.pkl")
# %%
seq = ["hello, nice to meet to", "how are you today"]
# %%
train = text_processor.process(seq, max_len=20, start_token=True, end_token=True)
# %%
text_processor.tokenizer.token_index
# %%
