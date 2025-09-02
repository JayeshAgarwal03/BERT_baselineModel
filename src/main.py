import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models import MultiTaskClassifier
from src.utils.data_parser import load_data, preprocess_data
from src.metrics.evaluator import evaluate_model, print_metrics
import sys
import os
sys.path.append('/content/BERT_baselineModel')


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess data
    examples, label_map = load_data("/content/dev_testset.json")
    train_ds, test_ds = preprocess_data(examples)

    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=16)

    # Initialize model
    model_name = "bert-base-uncased"
    model = MultiTaskClassifier(model_name, num_labels=3).to(device)

    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    EPOCHS = 7

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
