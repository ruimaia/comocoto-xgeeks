import json
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer

ROOT_DIR = Path(__file__).parent.parent.parent


if __name__ == "__main__":
    with open(ROOT_DIR / "data" / "historic_data.json") as f:
        data = json.load(f)
        
    caracteristics = [" ".join(datapoint["caracteristics"]) for datapoint in data]
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 4))
    
    vectorized_caracteristics = vectorizer.fit_transform(caracteristics)
    
    print(vectorized_caracteristics)