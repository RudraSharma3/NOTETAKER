from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

class Summarizer:
    def __init__(self, model_name="t5-small", device=None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)

    def summarize(self, text, max_length=128):
        input_text = "summarize: " + text
        inputs = self.tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        out = self.model.generate(inputs, max_length=max_length, num_beams=4, early_stopping=True)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)
