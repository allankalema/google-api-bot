import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class FAQRetriever:
    def __init__(self, path='faqs.xlsx'):
        self.df = pd.read_excel(path)
        self.questions = self.df['Questions'].astype(str).tolist()
        self.answers = self.df['Answers'].astype(str).tolist()
        self.vectorizer = TfidfVectorizer()
        self.q_vectors = self.vectorizer.fit_transform(self.questions)

    def retrieve(self, query, top_k=1):
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.q_vectors).flatten()
        top_indices = scores.argsort()[::-1][:top_k]
        context = [f"Q: {self.questions[i]}\nA: {self.answers[i]}" for i in top_indices]
        return "\n".join(context)
