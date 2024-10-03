import tiktoken
import torch

from config import gpt2_config
from data_loader.spam_loader import train_loader, val_loader, test_loader
from evaluate.batch_loss import calc_loss_loader
from evaluate.evaluate_classifier import calc_accuracy_loader, calc_classifier_loss_batch
from evaluate.evaluate_model import evaluate_model
from generate import generate_and_print_sample
from gpt.model import GPTModel
from parameters.load_gpt2 import load_gpt2
from parameters.load_weights import load_gpt_weights


def train_classifier_simple(model, train_loader, val_loader, optimizer, device,
                       num_epochs, eval_freq, eval_iter,
                       start_context,
                       tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_classifier_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    calc_classifier_loss_batch,
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch + 1} (Step {global_step:06d}):"
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen


def main(gpt_config, input_prompt, model_size):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    settings, params = load_gpt2(label=model_size, models_dir="gpt2")

    gpt = GPTModel(gpt_config)
    load_gpt_weights(gpt, params)
    gpt.to(device)
    gpt.eval()

    print(gpt)

    # Freeze the model layers
    for param in gpt.parameters():
        param.requires_grad = False

    # Replace the output layer with two node classifier layer
    num_classes = 2
    gpt.out_head = torch.nn.Linear(in_features=gpt_config.emb_dim, out_features=num_classes, bias=False)

    # Unfreeze the last Transformer layer and the final normalization layer for fine tuning
    for param in gpt.trf_blocks[-1].parameters():
        param.requires_grad = True
    for param in gpt.final_norm.parameters():
        param.requires_grad = True

    print(gpt.out_head)
    print(gpt.trf_blocks[-1])

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

    # Report the accuracy of the model before tuning
    train_accuracy = calc_accuracy_loader(train_loader, gpt, device, num_batches=10)
    val_accuracy = calc_accuracy_loader(val_loader, gpt, device, num_batches=10)
    test_accuracy = calc_accuracy_loader(test_loader, gpt, device, num_batches=10)

    print(f"Training accuracy: {train_accuracy*100:.2f}%")
    print(f"Validation accuracy: {val_accuracy*100:.2f}%")
    print(f"Test accuracy: {test_accuracy*100:.2f}%")

    train_loss = calc_loss_loader(calc_classifier_loss_batch, train_loader, gpt, device, num_batches=5)
    val_loss = calc_loss_loader(calc_classifier_loss_batch, val_loader, gpt, device, num_batches=5)
    test_loss = calc_loss_loader(calc_classifier_loss_batch, test_loader, gpt, device, num_batches=5)
    print(f"Training loss: {train_loss:.3f}")
    print(f"Validation loss: {val_loss:.3f}")
    print(f"Test loss: {test_loss:.3f}")


if __name__ == "__main__":

    torch.manual_seed(123)

    INPUT_PROMPT = "Do you have time"

    cfg, label = gpt2_config("small")

    main(cfg, INPUT_PROMPT, label)

