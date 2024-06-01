import tiktoken
import torch

from config import Context, GPT_CONFIG_124M
from gpt.model import GPTModel

tokenizer = tiktoken.get_encoding("gpt2")

batch = []

txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print(batch)

torch.manual_seed(123)
model = GPTModel(Context(GPT_CONFIG_124M))

out = model(batch)
print("Input batch:\n", batch)
print("Output shape:", out.shape)