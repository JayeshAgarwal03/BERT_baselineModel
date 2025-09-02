import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
sys.path.append('/content/BERT_baselineModel/src')

from models import MultiTaskClassifier
from utils.data_parser import load_data, preprocess_data
from metrics.evaluator import evaluate_model, print_metrics
from config.config import (
    MODEL_NAME, NUM_LABELS, EPOCHS, LEARNING_RATE, 
    BATCH_SIZE, MAX_LENGTH, TEST_SIZE, RANDOM_SEED, DROPOUT_RATE
)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess data
    examples, label_map = load_data("/content/dev_testset.json")
    train_ds, test_ds = preprocess_data(examples, MODEL_NAME, MAX_LENGTH, TEST_SIZE, RANDOM_SEED)

    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # Initialize model
    model = MultiTaskClassifier(MODEL_NAME, num_labels=NUM_LABELS, dropout_rate=DROPOUT_RATE).to(device)

    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            mistake_labels = batch["mistake"].to(device)
            guidance_labels = batch["guidance"].to(device)

            mistake_logits, guidance_logits = model(input_ids, attention_mask)
            loss = (
                criterion(mistake_logits, mistake_labels)
                + criterion(guidance_logits, guidance_labels)
            ) / 2
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)

        # Evaluation
        metrics = evaluate_model(model, test_loader, device)
        print_metrics(epoch, avg_train_loss, metrics)


if __name__ == "__main__":
    main()
