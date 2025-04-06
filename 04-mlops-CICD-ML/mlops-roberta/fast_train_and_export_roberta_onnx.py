from datasets import load_dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader
import torch
from torch.optim import AdamW
import onnx
import onnxruntime
import numpy as np

# 🔹 Load SST-2 dataset (only 300 samples for speed)
print("🔹 Loading SST-2 subset (300 samples)...")
dataset = load_dataset("glue", "sst2")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def tokenize(example):
    return tokenizer(example["sentence"], padding="max_length", truncation=True, max_length=128)

encoded_dataset = dataset.map(tokenize, batched=True)
encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

train_dataset = encoded_dataset["train"].select(range(300))  # <-- speed boost here
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# 🔹 Load RoBERTa model for classification
print("🔹 Initializing model...")
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

# 🔹 Training settings
optimizer = AdamW(model.parameters(), lr=5e-5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

# 🔹 Train for 1 epoch, 1 batch only
print("🔹 Training (only 1 batch)...")
for i, batch in enumerate(train_loader):
    optimizer.zero_grad()
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["label"].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()

    print(f"✅ Batch {i+1} complete - Loss: {loss.item():.4f}")
    break  # <-- Train only 1 batch

# 🔹 Export to ONNX
print("🔹 Exporting ONNX model...")
model.eval()

sample_text = "MLOps is amazing!"
tokens = tokenizer(sample_text, return_tensors="pt", padding="max_length", max_length=128, truncation=True)
input_ids = tokens["input_ids"]
attention_mask = tokens["attention_mask"]

torch.onnx.export(
    model,
    (input_ids, attention_mask),
    "roberta-sequence-classification-9.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["output"],
    dynamic_axes={"input_ids": {0: "batch_size"}, "attention_mask": {0: "batch_size"}},
    opset_version=11
)

print("✅ Exported to roberta-sequence-classification-9.onnx")

# 🔹 Quick test using ONNX Runtime
print("🔹 Testing inference with ONNX model...")
session = onnxruntime.InferenceSession("roberta-sequence-classification-9.onnx")

inputs = {
    "input_ids": input_ids.numpy(),
    "attention_mask": attention_mask.numpy()
}
outputs = session.run(None, inputs)
prediction = np.argmax(outputs[0])
label = "positive" if prediction == 1 else "negative"

print(f"✅ ONNX Inference Result: {label}")
