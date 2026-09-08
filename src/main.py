"""
SentimentScope: end-to-end entry point.

Trains a from-scratch transformer (DemoGPT) on the IMDB dataset for
binary sentiment classification, wiring together the modules/ package.

Expects the extracted IMDB dataset (aclImdb_v1/aclImdb/{train,test}/{pos,neg})
to already be present relative to this script.
"""

from modules import config
from modules.data_utils import build_dataframes, train_val_split
from modules.dataset import build_datasets, build_dataloaders
from modules.model import DemoGPT
from modules.train import calculate_accuracy, train_model


def main():
    # 1. Load and prepare data
    train_df, test_df = build_dataframes(
        config.TRAIN_POS_PATH,
        config.TRAIN_NEG_PATH,
        config.TEST_POS_PATH,
        config.TEST_NEG_PATH,
    )
    train_data, val_data = train_val_split(train_df)

    # 2. Build datasets / dataloaders
    train_dataset, val_dataset, test_dataset = build_datasets(
        train_data, val_data, test_df, config.tokenizer
    )
    train_loader, val_loader, test_loader = build_dataloaders(
        train_dataset, val_dataset, test_dataset, batch_size=config.BATCH_SIZE
    )

    # 3. Build model
    model = DemoGPT(config.config).to(config.device)

    # 4. Sanity check: accuracy before training (should be ~50%)
    baseline_accuracy = calculate_accuracy(model, val_loader, config.device)
    print(f"Baseline Validation Accuracy (untrained): {baseline_accuracy:.2f}%")

    # 5. Train
    model = train_model(
        model,
        train_loader,
        val_loader,
        config.device,
        epochs=config.EPOCHS,
        learning_rate=config.LEARNING_RATE,
    )

    # 6. Test
    test_accuracy = calculate_accuracy(model, test_loader, config.device)
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Goal met (>= 75%): {test_accuracy >= 75.0}")


if __name__ == "__main__":
    main()
