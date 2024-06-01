from parameters.load_gpt2 import load_gpt2

settings, params = load_gpt2("124M", "gpt2")
print(settings, params)