import pandas as pd

from spam import spam_data_file_path


# Balance the dataset with undersampling to the smaller subset. Map the classification values to 1/0
def create_balanced_dataset(df, label, yes_val, no_val):
    yes_subset = df[df[label] == yes_val]
    no_subset = df[df[label] == no_val]
    num_yes = len(yes_subset)
    num_no = len(no_subset)

    if num_yes > num_no:
        yes_subset = yes_subset.sample(num_no, random_state=123)  # Pin the random state
    else:
        no_subset = no_subset.sample(num_yes, random_state=123)  # Pin the random state

    balanced_df = pd.concat([yes_subset, no_subset])
    balanced_df[label] = balanced_df[label].map({yes_val: 1, no_val: 0})
    return balanced_df


def random_split(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    assert train_ratio + val_ratio + test_ratio == 1
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)  # Pin the random state

    train_end = int(len(df) * train_ratio)
    val_end = train_end + int(len(df) * val_ratio)

    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]
    return train_df, val_df, test_df


if __name__ == '__main__':
    sdf = pd.read_csv(spam_data_file_path, sep='\t', header=None, names=['Label', 'Text'])
    balanced_data = create_balanced_dataset(sdf, 'Label', "spam", "ham")
    print(balanced_data['Label'].value_counts())
    xtrain_df, xval_df, xtest_df = random_split(balanced_data, 0.7, 0.1, 0.2)

    xtrain_df.to_csv('train.csv', index=False)
    xval_df.to_csv('validation.csv', index=False)
    xtest_df.to_csv('test.csv', index=False)
