from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class SentimentIntentDemo:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
      

    def predict(self, text):
        toks = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            out = self.model(**toks)
        logits = out.logits
        probs = torch.softmax(logits, dim=1).squeeze().tolist()
        pred = int(torch.argmax(logits, dim=1))
        label = "Positive" if pred == 1 else "Negative"
        # Heuristic mapping:
        if label == "Negative":
            mapped = "Anxious"
        else:
            mapped = "Reassured"
        return {"raw_label": label, "mapped_sentiment": mapped, "scores": probs}

# Intent detection baseline (rule-based)
def detect_intent(text):
    txt = text.lower()
    if any(w in txt for w in ["worried", "concerned", "anxious", "scared", "nervous"]):
        return "Seeking reassurance"
    if any(w in txt for w in ["i had", "i was", "i have", "i experienced", "my neck", "my back"]):
        return "Reporting symptoms"
    return "Neutral/No specific intent"
