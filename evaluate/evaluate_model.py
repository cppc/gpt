import torch

from evaluate.batch_loss import calc_loss_loader


def evaluate_model(model, train_loader, val_loader, batch_loss_calc, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(batch_loss_calc, train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(batch_loss_calc, val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss
