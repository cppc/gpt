import tiktoken
import torch
from matplotlib import pyplot as plt

from config import Context, GPT_CONFIG_124M_256
from data_loader import TEXT_DATA, create_dataloader_v1
from evaluate.batch_loss import calc_loss_batch
from evaluate.evaluate_model import evaluate_model
from generate import generate_and_print_sample
from gpt.model import GPTModel


def train_model_simple(model, train_loader, val_loader, optimizer, device,
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
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    calc_loss_batch,
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


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots()

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.show()


def main(cfg, settings):
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GPTModel(cfg)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay
    )

    train_ratio = 0.90
    split_idx = int(train_ratio * len(TEXT_DATA))

    train_loader = create_dataloader_v1(
        TEXT_DATA[:split_idx],
        batch_size=settings.batch_size,
        max_length=cfg.context_length,
        stride=cfg.context_length,
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = create_dataloader_v1(
        TEXT_DATA[split_idx:],
        batch_size=settings.batch_size,
        max_length=cfg.context_length,
        stride=cfg.context_length,
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    tokenizer = tiktoken.get_encoding("gpt2")

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=settings.num_epochs, eval_freq=5, eval_iter=1,
        start_context="Every effort moves you", tokenizer=tokenizer
    )

    return train_losses, val_losses, tokens_seen, model, optimizer


if __name__ == "__main__":
    cfg = Context(GPT_CONFIG_124M_256)
    settings = Context({
        "learning_rate": 5e-4,
        "num_epochs": 10,
        "batch_size": 2,
        "weight_decay": 0.1
    })

    train_losses, val_losses, tokens_seen, model, optimizer = main(cfg, settings)

    epochs_tensor = torch.linspace(0, settings.num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    plt.savefig("loss.pdf")

    torch.save(model.state_dict(), "model.pth")
    model = GPTModel(cfg)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    },
        "model_and_optimizer.pth")
