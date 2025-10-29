from typing import Dict, List

def map_to_soap(transcript_turns, ner_result, summary_text):
    """
    transcript_turns: list of (speaker, text)
    ner_result: output from ner.extract_medical_info
    summary_text: summarized narrative
    Returns SOAP JSON structure.
    """
    # Subjective: collect patient turns about symptoms/history
    subj_lines = []
    for sp, txt in transcript_turns:
        if sp.lower().startswith("patient"):
            subj_lines.append(txt)
    subjective = {
        "Chief_Complaint": ", ".join(ner_result.get("Symptoms", [])) or "Not explicitly stated",
        "History_of_Present_Illness": summary_text
    }
    # Objective: look for physician exam lines
    obj_lines = []
    for sp, txt in transcript_turns:
        if sp.lower().startswith("physician") and any(w in txt.lower() for w in ["examination","range of motion","tender","observed","looks good"]):
            obj_lines.append(txt)
    objective = {
        "Physical_Exam": " ".join(obj_lines) if obj_lines else "No structured physical exam text found in transcript.",
        "Observations": None
    }
    assessment = {
        "Diagnosis": ner_result.get("Diagnosis") or "Not found",
        "Severity": "Improving" if "improv" in (summary_text.lower() if summary_text else "") else "Not specified"
    }
    plan = {
        "Treatment": ner_result.get("Treatment", []),
        "Follow_Up": "Return if symptoms worsen or persist beyond expected recovery"
    }
    return {
        "Subjective": subjective,
        "Objective": objective,
        "Assessment": assessment,
        "Plan": plan
    }
