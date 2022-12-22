import requests
from transformers import pipeline, AutoTokenizer, AutoModel, AutoModelForSequenceClassification, Trainer
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

def getReviewsSentiments(uid):
    response = requests.get(f'https://api.jikan.moe/v4/anime/{uid}/reviews')
    path = "model/siebert/sentiment-roberta-large-english"
    data =response.json()
    pred_texts = [data.get('data')[0].get('review'),data.get('data')[1].get('review'),data.get('data')[2].get('review')]

    # Create class for data preparation
    class SimpleDataset:
        def __init__(self, tokenized_texts):
            self.tokenized_texts = tokenized_texts

        def __len__(self):
            return len(self.tokenized_texts["input_ids"])

        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self.tokenized_texts.items()}

    # Load tokenizer and model, create trainer
    model_name = "model/siebert/sentiment-roberta-large-english"
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True)
    trainer = Trainer(model=model)

    # Tokenize texts and create prediction data set
    tokenized_texts = tokenizer(pred_texts,truncation=True,padding=True)
    pred_dataset = SimpleDataset(tokenized_texts)

    # Run predictions
    predictions = trainer.predict(pred_dataset)

    # Transform predictions to labels
    preds = predictions.predictions.argmax(-1)
    labels = pd.Series(preds).map(model.config.id2label)
    scores = (np.exp(predictions[0])/np.exp(predictions[0]).sum(-1,keepdims=True)).max(1)

    # Create DataFrame with texts, predictions, labels, and scores
    df = pd.DataFrame(list(zip(pred_texts,labels,preds)), columns=['text','label','pred'])
    return df

