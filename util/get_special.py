import tiktoken


def get_special_token(token, encoding):
    tokenizer = tiktoken.get_encoding(encoding)
    token_code = tokenizer.encode(token, allowed_special={token})
    return token_code[0]
