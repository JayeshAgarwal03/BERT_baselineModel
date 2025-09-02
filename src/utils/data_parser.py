import json
from datasets import Dataset as HFDataset
from transformers import AutoTokenizer


def load_data(file_path: str):
    """Load data from JSON file and convert to HuggingFace dataset format."""
    with open(file_path, "r") as f:
        raw_data = json.load(f)
    
    label_map = {"Yes": 0, "No": 1, "To some extent": 2}
    
    examples = []
    for ex in raw_data:
        history = ex["conversation_history"]
        for tutor_name, tutor_entry in ex["tutor_responses"].items():
            response_text = tutor_entry["response"]
            ann = tutor_entry["annotation"]
            if ann:
                examples.append({
                    "history": history,
                    "response": response_text,
                    "mistake": label_map[ann["Mistake_Identification"]],
                    "guidance": label_map[ann["Providing_Guidance"]],
                })
    
    return examples, label_map


def preprocess_data(examples, model_name: str = "bert-base-uncased"):
    """Preprocess data using tokenizer and split into train/test sets."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def preprocess(batch):
        return tokenizer(
            batch["history"],
            batch["response"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )
    
    hf_dataset = HFDataset.from_list(examples)
    hf_dataset = hf_dataset.train_test_split(test_size=0.2, seed=42)
    train_ds, test_ds = hf_dataset["train"], hf_dataset["test"]
    
    # Apply preprocessing
    train_ds = train_ds.map(preprocess, batched=True, remove_columns=["history", "response"])
    test_ds = test_ds.map(preprocess, batched=True, remove_columns=["history", "response"])
    
    # Set format for PyTorch
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "mistake", "guidance"])
    test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "mistake", "guidance"])
    
    return train_ds, test_ds
