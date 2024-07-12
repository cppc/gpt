import torch
from torch.utils.data import DataLoader

from instructions.dataset import train_dataset, val_dataset, test_dataset

num_workers = 0
batch_size = 8
torch.manual_seed(123)  # Pin random value

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True
)


if __name__ == '__main__':
    for input_batch, target_batch in train_loader:
        pass
    print("Input batch shape:", input_batch.shape)
    print("Label batch shape:", target_batch.shape)
    print(f"{len(train_loader)} train batches")
    print(f"{len(val_loader)} validation batches")
    print(f"{len(test_loader)} test batches")
