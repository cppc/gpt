import tiktoken
import torch

from config import gpt2_config
from generate import text_to_token_ids, token_ids_to_text
from generate.generator import generate
from gpt.model import GPTModel
from parameters.load_gpt2 import load_gpt2
from parameters.load_weights import load_gpt_weights


def main(gpt_config, input_prompt, model_size):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    settings, params = load_gpt2(label=model_size, models_dir="gpt2")

    gpt = GPTModel(gpt_config)
    load_gpt_weights(gpt, params)
    gpt.to(device)
    gpt.eval()

    tokenizer = tiktoken.get_encoding("gpt2")

    token_ids = generate(
        model=gpt,
        idx=text_to_token_ids(input_prompt, tokenizer),
        max_new_tokens=50,
        context_size=gpt_config["context_length"],
        top_k=1,
        temperature=1.0
    )

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


if __name__ == "__main__":

    torch.manual_seed(123)

    INPUT_PROMPT = "Why does the sky look blue?"

    cfg, label = gpt2_config("small")

    main(cfg, INPUT_PROMPT, label)