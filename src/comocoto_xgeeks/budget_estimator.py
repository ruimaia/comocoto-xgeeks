import json
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer

ROOT_DIR = Path(__file__).parent.parent.parent


if __name__ == "__main__":
    with open(ROOT_DIR / "data" / "historic_data.json") as f:
        data = json.load(f)
        
    characteristics = [" ".join(datapoint["characteristics"]) for datapoint in data]
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 4))
    
    vectorized_characteristics = vectorizer.fit_transform(characteristics)
    X = vectorized_characteristics.toarray()
    
    print(X)