import tiktoken


def encode_special(token, encoding="gpt2"):
    tokenizer = tiktoken.get_encoding(encoding)
    return tokenizer.encode(token, allowed_special=token)
