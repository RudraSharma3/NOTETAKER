from collections import defaultdict
import spacy


try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


CATEGORY_KEYWORDS = {
    "Diagnosis": ["whiplash", "concussion", "fracture", "sprain", "strain"],
    "Treatment": ["physiotherapy", "physio", "painkillers", "analgesic", "x-ray", "xray", "xrays"],
    "Symptoms": ["neck pain", "back pain", "headache", "stiffness", "dizziness"],
    "Prognosis": ["full recovery", "recovery", "no long-term", "no lasting damage", "expect you to make a full recovery"]
}

def keyword_scan(text):
    txt_lower = text.lower()
    found = defaultdict(list)
    for cat, kwlist in CATEGORY_KEYWORDS.items():
        for kw in kwlist:
            if kw in txt_lower:
                found[cat].append(kw)
    return dict(found)

def spacy_entities(text):
    doc = nlp(text)
    ents = []
    for ent in doc.ents:
        ents.append({"text": ent.text, "label": ent.label_})
    return ents

def extract_medical_info(transcript_text):
    """
    Returns dict with extracted fields using both keyword scanning and spaCy.
    Also returns confidence flags for heuristic outputs.
    """
    scan = keyword_scan(transcript_text)
    ents = spacy_entities(transcript_text)
    result = {
        "raw_keywords": scan,
        "entities": ents,
    }
    # Heuristic summarization for expected fields
    symptoms = list(set(scan.get("Symptoms", [])))
    diagnosis = scan.get("Diagnosis", [])
    treatment = list(set(scan.get("Treatment", [])))
    prognosis = list(set(scan.get("Prognosis", [])))

    # person/date extraction
    patient_name = None
    dates = []
    for e in ents:
        if e['label'] == "PERSON" and not patient_name:
            patient_name = e['text']
        if e['label'] == "DATE":
            dates.append(e['text'])

    result.update({
        "Patient_Name": patient_name,
        "Symptoms": symptoms,
        "Diagnosis": diagnosis[0] if diagnosis else None,
        "Treatment": treatment,
        "Prognosis": prognosis,
        "Dates": dates
    })
    return result
