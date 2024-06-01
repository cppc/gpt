import re

from tokenizer.simple_tokenizer_v1 import SimpleTokenizerV1
from tokenizer.simple_tokenizer_v2 import SimpleTokenizerV2

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    print("Total characters:", len(raw_text))
    print(raw_text[:99])

    preprocessed = re.split(r'([,.?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]

    all_tokens = sorted(list(set(preprocessed)))
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])

    vocab = {token: integer for integer, token in enumerate(all_tokens)}

    tokenizer = SimpleTokenizerV2(vocab)

    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."

    text = " <|endoftext|> ".join((text1, text2))

    print(text)    # text = """"It's the last he painted, you know," Mrs. Gisburn said with pardonable pride."""
    ids = tokenizer.encode(text)
    print(ids)

    print(tokenizer.decode(tokenizer.encode(text)))