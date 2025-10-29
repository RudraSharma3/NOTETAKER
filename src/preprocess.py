import re
from typing import List, Tuple

def split_turns(transcript_text: str) -> List[Tuple[str,str]]:
    """
    Splits transcript by labelled turns like 'Physician:' and 'Patient:'.
    Returns list of (speaker, text).
    """
    # Normalize newlines
    text = transcript_text.strip()
    # Use regex to split at each newline that precedes a speaker tag
    turns = re.split(r'\n(?=(Physician:|Patient:))', text)
    out = []
    # Rejoin split pieces properly
    combined = []
    i = 0
    while i < len(turns):
        chunk = turns[i]
        if chunk in ("Physician:", "Patient:"):
            # next element exists
            if i+1 < len(turns):
                combined.append(chunk + " " + turns[i+1].strip())
                i += 2
            else:
                combined.append(chunk)
                i += 1
        else:
            combined.append(chunk)
            i += 1
    for c in combined:
        if ':' in c:
            sp, txt = c.split(':', 1)
            out.append((sp.strip(), txt.strip()))
    return out

def get_patient_turns(turns):
    return [t for t in turns if t[0].lower().startswith("patient")]
