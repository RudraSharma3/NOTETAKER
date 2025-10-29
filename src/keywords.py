from rake_nltk import Rake

def extract_keywords(text, top_n=10):
    r = Rake()
    r.extract_keywords_from_text(text)
    return r.get_ranked_phrases()[:top_n]
