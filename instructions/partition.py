
def partition_dataset(data, train_pct=0.85, test_pct=0.1):
    train_portion = int(len(data) * train_pct)
    test_portion = int(len(data) * test_pct)
    val_portion = len(data) - train_portion - test_portion
    return data[:train_portion], data[train_portion:train_portion + test_portion], data[train_portion + test_portion:]
