import json
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from datasets import Dataset as HFDataset
import evaluate
from tqdm import tqdm

from src.models import MultiTaskClassifier


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("/content/dev_testset.json", "r") as f:
        raw_data = json.load(f)

    label_map = {"Yes": 0, "No": 1, "To some extent": 2}

    examples = []
    for ex in raw_data:
        history = ex["conversation_history"]
        for tutor_name, tutor_entry in ex["tutor_responses"].items():
            response_text = tutor_entry["response"]
            ann = tutor_entry["annotation"]
            if ann:
                # CHANGE 1: Store history and response in separate keys instead of one "text" key.
                examples.append({
                    "history": history,
                    "response": response_text,
                    "mistake": label_map[ann["Mistake_Identification"]],
                    "guidance": label_map[ann["Providing_Guidance"]],
                })

    hf_dataset = HFDataset.from_list(examples)
    hf_dataset = hf_dataset.train_test_split(test_size=0.2, seed=42)
    train_ds, test_ds = hf_dataset["train"], hf_dataset["test"]

    # =========================
    # 2. Tokenizer & Encoding (MODIFIED)
    # =========================
    model_name = "bert-base-uncased"  # Other models: bert-base-uncased, t5-base, facebook/bart-base
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # CHANGE 2: Update the preprocess function to tokenize the history and response as a pair.
    def preprocess(batch):
        # The tokenizer will now automatically format the input as:
        # [CLS] history [SEP] response [SEP]
        return tokenizer(
            batch["history"],
            batch["response"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )

    # The .map function now applies the new preprocess function.
    # We also remove the original text columns as they are no longer needed after tokenization.
    train_ds = train_ds.map(preprocess, batched=True, remove_columns=["history", "response"])
    test_ds = test_ds.map(preprocess, batched=True, remove_columns=["history", "response"])

    # This part remains the same as it correctly selects the columns needed for training.
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "mistake", "guidance"])
    test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "mistake", "guidance"])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=16)

    # =========================
    # 3. Multi-task Model
    # =========================
    model = MultiTaskClassifier(model_name, num_labels=3).to(device)

    # =========================
    # 4. Training Setup
    # =========================
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    metric = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")

    EPOCHS = 7

    # =========================
    # 5. Training Loop
    # =========================
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

        # =========================
        # 6. Evaluation
        # =========================
        model.eval()
        mistake_preds, mistake_refs = [], []
        guidance_preds, guidance_refs = [], []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                mistake_labels = batch["mistake"].to(device)
                guidance_labels = batch["guidance"].to(device)

                mistake_logits, guidance_logits = model(input_ids, attention_mask)
                mistake_preds.extend(mistake_logits.argmax(dim=-1).cpu().tolist())
                guidance_preds.extend(guidance_logits.argmax(dim=-1).cpu().tolist())
                mistake_refs.extend(mistake_labels.cpu().tolist())
                guidance_refs.extend(guidance_labels.cpu().tolist())

        mistake_acc = metric.compute(predictions=mistake_preds, references=mistake_refs)["accuracy"]
        guidance_acc = metric.compute(predictions=guidance_preds, references=guidance_refs)["accuracy"]

        # The `average="macro"` argument is key. It tells the metric to calculate F1 for each class
        # and then find the unweighted average.
        mistake_f1 = metric_f1.compute(predictions=mistake_preds, references=mistake_refs, average="macro")["f1"]
        guidance_f1 = metric_f1.compute(predictions=guidance_preds, references=guidance_refs, average="macro")["f1"]

        print(
            f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | "
            f"Mistake Acc: {mistake_acc:.4f} | Mistake Macro-F1: {mistake_f1:.4f} | "
            f"Guidance Acc: {guidance_acc:.4f} | Guidance Macro-F1: {guidance_f1:.4f}"
        )


if __name__ == "__main__":
    main()
