import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

device = torch.device("cpu")

model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

class StreamingTextDataset(Dataset):
    def __init__(self, filepath, tokenizer, block_size=128):
        self.examples = []
        with open(filepath, "r", encoding="utf-8") as f:
            buffer = ""
            for line in f:
                buffer += line.strip() + " "
                if len(buffer) > block_size * 10:
                    tokens = tokenizer(buffer, truncation=True, padding="max_length", max_length=block_size)
                    self.examples.append({
                        "input_ids": torch.tensor(tokens["input_ids"]),
                        "attention_mask": torch.tensor(tokens["attention_mask"]),
                        "labels": torch.tensor(tokens["input_ids"]),
                    })
                    buffer = ""

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

dataset = StreamingTextDataset("estate.txt", tokenizer)

training_args = TrainingArguments(
    output_dir="./slm_cpu_model",
    per_device_train_batch_size=4,
    num_train_epochs=10,
    learning_rate=1e-5,
    logging_steps=10,
    save_steps=100,
    save_strategy="epoch",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()
model.save_pretrained("./my_trained_model")
tokenizer.save_pretrained("./my_trained_model")
