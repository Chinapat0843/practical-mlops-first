from datasets import load_dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader
import torch
from torch.optim import AdamW
import onnxruntime
import numpy as np
import torch.onnx

print("🔹 Loading SST-2 subset (300 samples)...")
dataset = load_dataset("glue", "sst2")
tokenizer = RobertaTokenizer.from_pretrained("distilroberta-base")

def tokenize(example):
    return tokenizer(example["sentence"], padding="max_length", truncation=True, max_length=128)

encoded_dataset = dataset.map(tokenize, batched=True)
encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

train_dataset = encoded_dataset["train"].select(range(300))
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

print("🔹 Initializing distilroberta-base model...")
model = RobertaForSequenceClassification.from_pretrained("distilroberta-base", num_labels=2)

optimizer = AdamW(model.parameters(), lr=5e-5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

print("🔹 Training (1 batch only for speed)...")
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
    break

print("🔹 Exporting to ONNX...")
model.eval()

sample_text = "MLOps is amazing!"
tokens = tokenizer(sample_text, return_tensors="pt", padding="max_length", max_length=128, truncation=True)
input_ids = tokens["input_ids"]
attention_mask = tokens["attention_mask"]

torch.onnx.export(
    model,
    (input_ids, attention_mask),
    "distilroberta-sequence-classification.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["output"],
    dynamic_axes={"input_ids": {0: "batch_size"}, "attention_mask": {0: "batch_size"}},
    opset_version=11
)

print("✅ Exported ONNX: distilroberta-sequence-classification.onnx")

print("🔹 Testing ONNX inference...")
session = onnxruntime.InferenceSession("distilroberta-sequence-classification.onnx")

inputs = {
    "input_ids": input_ids.numpy(),
    "attention_mask": attention_mask.numpy()
}
outputs = session.run(None, inputs)
prediction = np.argmax(outputs[0])
label = "positive" if prediction == 1 else "negative"

print(f"✅ ONNX Inference: '{sample_text}' → {label}")
