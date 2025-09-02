import torch
from torch.utils.data import DataLoader
import evaluate
from tqdm import tqdm


def evaluate_model(model, test_loader, device):
    """Evaluate the model on test data and return metrics."""
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

    # Calculate metrics
    metric = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")
    
    mistake_acc = metric.compute(predictions=mistake_preds, references=mistake_refs)["accuracy"]
    guidance_acc = metric.compute(predictions=guidance_preds, references=guidance_refs)["accuracy"]
    
    mistake_f1 = metric_f1.compute(predictions=mistake_preds, references=mistake_refs, average="macro")["f1"]
    guidance_f1 = metric_f1.compute(predictions=guidance_preds, references=guidance_refs, average="macro")["f1"]
    
    return {
        "mistake_acc": mistake_acc,
        "guidance_acc": guidance_acc,
        "mistake_f1": mistake_f1,
        "guidance_f1": guidance_f1
    }


def print_metrics(epoch, avg_train_loss, metrics):
    """Print formatted metrics."""
    print(
        f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | "
        f"Mistake Acc: {metrics['mistake_acc']:.4f} | Mistake Macro-F1: {metrics['mistake_f1']:.4f} | "
        f"Guidance Acc: {metrics['guidance_acc']:.4f} | Guidance Macro-F1: {metrics['guidance_f1']:.4f}"
    )
