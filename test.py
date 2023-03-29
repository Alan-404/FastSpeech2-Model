#%%
import torch
# %%
a = torch.rand((10, 512, 20))
# %%
batch = torch.nn.BatchNorm1d(num_features=512)
# %%
b = batch(a)
# %%
b.size()
# %%
