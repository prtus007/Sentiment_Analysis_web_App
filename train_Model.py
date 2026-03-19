import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

texts = []
labels = []

files = [
    "data/amazon_cells_labelled.txt",
    "data/imdb_labelled.txt",
    "data/yelp_labelled.txt"
]

for file_path in files:
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                texts.append(parts[0])
                labels.append(int(parts[1]))

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model and vectorizer saved successfully.")
