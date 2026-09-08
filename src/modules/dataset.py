"""
PyTorch Dataset and DataLoader construction for the IMDB reviews.
"""

from torch.utils.data import Dataset, DataLoader

from modules.config import MAX_LENGTH, BATCH_SIZE


class IMDBDataset(Dataset):
    """
    A custom PyTorch Dataset for the IMDB dataset.

    This class preprocesses text data using a tokenizer and returns tokenized inputs
    along with their corresponding labels for sentiment analysis.
    """

    def __init__(self, data, tokenizer, max_length=MAX_LENGTH):
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]["review"]
        label = int(self.data.iloc[idx]["label"])

        tokens = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = tokens["input_ids"].squeeze(0)

        return input_ids, label


def build_datasets(train_data, val_data, test_df, tokenizer):
    """Wraps train/val/test DataFrames into IMDBDataset instances."""
    train_dataset = IMDBDataset(train_data, tokenizer)
    val_dataset = IMDBDataset(val_data, tokenizer)
    test_dataset = IMDBDataset(test_df, tokenizer)
    return train_dataset, val_dataset, test_dataset


def build_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=BATCH_SIZE):
    """Wraps datasets into DataLoaders (train shuffled, val/test not)."""
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
