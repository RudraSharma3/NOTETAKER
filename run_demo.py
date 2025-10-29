import json
from src.preprocess import split_turns, get_patient_turns
from src.ner import extract_medical_info
from src.keywords import extract_keywords
from src.summarizer import Summarizer
from src.sentiment_intent import SentimentIntentDemo, detect_intent
from src.soap import map_to_soap

def load_transcript(path="sample_inputs/transcript.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def main():
    transcript = load_transcript()
    turns = split_turns(transcript)
    
    # Concatenate patient turns for sentiment/intent
    patient_turns = [t for t in turns if t[0].lower().startswith("patient")]
    patient_text = " ".join([t[1] for t in patient_turns])

    # NER / medical extract
    ner_result = extract_medical_info(transcript)

    # Keywords
    keywords = extract_keywords(transcript)

    # Summarization
    summarizer = Summarizer(model_name="t5-small")
    summary = summarizer.summarize(transcript, max_length=150)

    # Sentiment & Intent (demo)
    senti = SentimentIntentDemo()
    sentiment = senti.predict(patient_text)
    intent = detect_intent(patient_text)

    # SOAP
    soap = map_to_soap(turns, ner_result, summary)

    out = {
        "NER": ner_result,
        "Keywords": keywords,
        "Summary": summary,
        "Sentiment": sentiment,
        "Intent": intent,
        "SOAP": soap
    }

    print(json.dumps(out, indent=2))
    with open("sample_outputs/result.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    main()
