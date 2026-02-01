import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Portfolio:
    def __init__(self, file_path="resource/my_portfolio.csv"):
        self.data = pd.read_csv(file_path)

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = None

    def load_portfolio(self):
        techstacks = self.data["Techstack"].tolist()
        self.tfidf_matrix = self.vectorizer.fit_transform(techstacks)

    def query_links(self, skills, k=2):
        if not skills:
            return []

        query_vec = self.vectorizer.transform([skills])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        top_indices = similarities.argsort()[-k:][::-1]
        return [{"links": self.data.iloc[i]["Links"]} for i in top_indices]
