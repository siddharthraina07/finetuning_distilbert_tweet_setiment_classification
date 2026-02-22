import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline
)
import evaluate
from sklearn.metrics import classification_report


dataset = load_dataset("tweet_eval", "sentiment")


model_checkpoint = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize_function(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

encoded_dataset = dataset.map(tokenize_function, batched=True)

encoded_dataset = encoded_dataset.remove_columns(["text"])
encoded_dataset = encoded_dataset.rename_column("label", "labels")
encoded_dataset.set_format("torch")

model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=3
)

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
        "f1": f1.compute(predictions=predictions, references=labels, average="weighted")["f1"]
    }

import os
os.environ["TENSORBOARD_LOGGING_DIR"] = "./logs"
training_args = TrainingArguments(
    output_dir="./distilbert-tweet-sentiment",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    compute_metrics=compute_metrics
)

trainer.train()

trainer.evaluate(encoded_dataset["test"])