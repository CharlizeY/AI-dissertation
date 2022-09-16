import re

import numpy as np
import torch
from datasets import Dataset, load_metric
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, DataCollatorWithPadding, \
    TrainingArguments, EarlyStoppingCallback

from concept_processing.io import get_datapoint_iterator
from sklearn.model_selection import train_test_split

torch.cuda.empty_cache()

# examples_dir = '../../full_dataset'
# num_labels = 5
# examples_dir = '/home/yy3219/projects/thesis/bird_flowers_ds'
num_labels = 2 # Change this
examples_dir = '/vol/bitbucket/yy3219/roko-for-charlize/2000_pn_dataset'
# examples_dir = '/Users/Cherry0904/Desktop/roko-for-charlize/short_dataset'
model_name = 'roberta-base'

ids, labels, texts = [], [], []
tokenizer_max_length = 160 
batch_size = 32


def get_data_collator(tokenizer):
    return DataCollatorWithPadding(tokenizer, max_length=tokenizer_max_length, padding='max_length')


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return load_metric("f1").compute(predictions=predictions, references=labels, average='macro')


def to_model_dataset(pandas_dataset, tokenizer):
    def to_model_dataset_single(pd_dataset):
        dataset = Dataset.from_pandas(pd_dataset)
        dataset = dataset.map(
            lambda example: tokenizer(example['text'], truncation=True, max_length=tokenizer_max_length, padding=True),
            batched=True)
        dataset = dataset.remove_columns(['text', '__index_level_0__'] )
        if "label" in dataset.features:
            dataset = dataset.rename_column("label", "labels")
        return dataset

    if type(pandas_dataset) is tuple:
        return tuple(map(to_model_dataset_single, pandas_dataset))

    return to_model_dataset_single(pandas_dataset)


def make_predictions(model, tokenizer, dataset, batch_size, device):
    data_collator = get_data_collator(tokenizer)
    dataloader = DataLoader(dataset, collate_fn=data_collator, batch_size=batch_size)
    # dataloader = DataLoader(dataset, batch_size=batch_size)
    y_pred = []
    y_true = []

    for i, batch in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs["logits"]
            preds = torch.argmax(logits, dim=-1)
            y_pred.extend(preds.tolist())  # Save Prediction

            if "labels" in batch:
                y_true.extend(batch["labels"].tolist())
            else:
                y_true = None

    return y_pred, y_true


if __name__ == "__main__":

    for id_, label, text in get_datapoint_iterator(examples_dir):
        try:
            # we remove double whitespace as it breaks the benepar plugin
            text = text.replace('�', '')
            text = re.sub(' +', ' ', text)
        except Exception:
            print(f"Error occurred because of text = {text}")
            raise

        ids.append(id_)
        labels.append(label)
        texts.append(text)

    # Model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(device)

    # Dataset
    df = pd.DataFrame.from_dict({"label": labels, "text": texts})
    df = df.drop(df[df.label == "none"].index)
    # df.label[
        # df.label == ' it could be called a strike because the pitch landed in the strike zone before being hit'] = 'strike'
    label_rev_dict = np.unique(df.label)

    for i, val in enumerate(label_rev_dict):
        df.label[df.label == val] = i

    train_df, test_df = train_test_split(df, train_size=1518) # Change the train size (0.85/0.15) 
    train_ds, test_ds = to_model_dataset((train_df, test_df), tokenizer)

    args = TrainingArguments(
        output_dir="training-output",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=6,
        lr_scheduler_type="cosine_with_restarts",
        learning_rate=1e-5,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    trainer = Trainer(model=model,
                      args=args,
                      train_dataset=train_ds,
                      eval_dataset=test_ds,
                      tokenizer=tokenizer,
                      data_collator=get_data_collator(tokenizer),
                      compute_metrics=compute_metrics,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
                      )

    trainer.train()
    model.eval()

    preds, true = make_predictions(model, tokenizer, test_ds, batch_size=4, device=device)
    conf_matrix = confusion_matrix(true, preds)
    f1 = f1_score(true, preds, average='macro')
    accuracy = accuracy_score(true, preds)

    print(f"Model finished with accuracy: {accuracy}, macro-f1: {f1}")
    print("Confusion matrix:")
    print(conf_matrix)
