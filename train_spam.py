import tiktoken
import torch

from config import gpt2_config
from data_loader.spam_loader import train_loader, val_loader, test_loader
from evaluate.evaluate_classifier import calc_accuracy_loader
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

    print(gpt)

    for param in gpt.parameters():
        param.requires_grad = False

    num_classes = 2
    gpt.out_head = torch.nn.Linear(in_features=gpt_config.emb_dim, out_features=num_classes)

    for param in gpt.trf_blocks[-1].parameters():
        param.requires_grad = True
    for param in gpt.final_norm.parameters():
        param.requires_grad = True

    tokenizer = tiktoken.get_encoding("gpt2")

    inputs = tokenizer.encode(input_prompt)
    inputs = torch.tensor(inputs).unsqueeze(0)
    print("Inputs:", inputs)
    print("Inputs dimensions:", inputs.shape)

    with torch.no_grad():
        outputs = gpt(inputs)
    print("Outputs:", outputs)
    print("Outputs dimensions:", outputs.shape)

    print("Last output token:", outputs[:, -1, :])

    train_accuracy = calc_accuracy_loader(train_loader, gpt, device, num_batches=10)
    val_accuracy = calc_accuracy_loader(val_loader, gpt, device, num_batches=10)
    test_accuracy = calc_accuracy_loader(test_loader, gpt, device, num_batches=10)

    print(f"Training accuracy: {train_accuracy*100:.2f}%")
    print(f"Validation accuracy: {val_accuracy*100:.2f}%")
    print(f"Test accuracy: {test_accuracy*100:.2f}%")


if __name__ == "__main__":

    torch.manual_seed(123)

    INPUT_PROMPT = "Do you have time"

    cfg, label = gpt2_config("small")

    main(cfg, INPUT_PROMPT, label)

