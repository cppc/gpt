import tiktoken

from config import Context, GPT_CONFIG_124M_256
from generate import generate_text_simple
from generate.util import text_to_token_ids, token_ids_to_text
from gpt.model import GPTModel

cfg = Context(GPT_CONFIG_124M_256)
model = GPTModel(cfg)
model.eval()
start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=cfg.context_length
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
