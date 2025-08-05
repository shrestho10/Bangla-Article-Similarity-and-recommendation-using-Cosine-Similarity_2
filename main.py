import os
import math
import re
from collections import defaultdict, Counter

# ==== CONFIG ====
ARTICLE_DIR = "articles"
STOPWORDS_FILE = "stopwords_bn.txt"
TOP_K = 5  # Number of recommended articles


# ==== UTILITY FUNCTIONS ====

def load_stopwords():
    with open(STOPWORDS_FILE, 'r', encoding='utf-8') as f:
        return set(f.read().split())

def tokenize(text):
    text = re.sub(r'[^\u0980-\u09FF\s]', '', text)  # keep only Bangla characters
    return text.strip().split()

def clean_tokens(tokens, stopwords):
    return [token for token in tokens if token not in stopwords and len(token) > 1]


# ==== CORE CLASS ====

class BanglaRecommender:
    def __init__(self, article_dir, stopword_file):
        self.article_dir = article_dir
        self.stopwords = load_stopwords()
        self.documents = []
        self.doc_names = []
        self.vocab = set()
        self.tf = []
        self.idf = {}
        self.tfidf = []

        self.load_documents()
        self.compute_tf()
        self.compute_idf()
        self.compute_tfidf()

    def load_documents(self):
        print("Loading documents...")
        for filename in sorted(os.listdir(self.article_dir)):
            if filename.endswith('.txt'):
                with open(os.path.join(self.article_dir, filename), 'r', encoding='utf-8') as f:
                    text = f.read()
                    tokens = tokenize(text)
                    tokens = clean_tokens(tokens, self.stopwords)
                    self.documents.append(tokens)
                    self.doc_names.append(filename)
        print(f"{len(self.documents)} articles loaded.")

    def compute_tf(self):
        print("Computing term frequency (TF)...")
        for doc in self.documents:
            freq = Counter(doc)
            total = len(doc)
            tf_doc = {word: freq[word] / total for word in freq}
            self.tf.append(tf_doc)
            self.vocab.update(tf_doc.keys())

    def compute_idf(self):
        print("Computing inverse document frequency (IDF)...")
        doc_count = defaultdict(int)
        total_docs = len(self.documents)
        for word in self.vocab:
            for doc in self.documents:
                if word in doc:
                    doc_count[word] += 1
        self.idf = {word: math.log(total_docs / (1 + doc_count[word])) for word in doc_count}

    def compute_tfidf(self):
        print("Computing TF-IDF...")
        for tf_doc in self.tf:
            tfidf_doc = {word: tf_doc[word] * self.idf[word] for word in tf_doc}
            self.tfidf.append(tfidf_doc)

    def vectorize(self, tokens):
        tokens = clean_tokens(tokens, self.stopwords)
        freq = Counter(tokens)
        total = len(tokens)
        tf = {word: freq[word] / total for word in freq}
        tfidf = {word: tf[word] * self.idf.get(word, 0) for word in tf}
        return tfidf

    def cosine_similarity(self, vec1, vec2):
        dot_product = sum(vec1.get(w, 0) * vec2.get(w, 0) for w in vec1)
        norm1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
        norm2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def recommend(self, query, top_k=TOP_K):
        print("\n[INFO] Recommending top", top_k, "articles for:", query)
        query_tokens = tokenize(query)
        query_vec = self.vectorize(query_tokens)
        scores = []
        for idx, doc_vec in enumerate(self.tfidf):
            score = self.cosine_similarity(query_vec, doc_vec)
            scores.append((self.doc_names[idx], score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ==== RUN CODE ====

if __name__ == "__main__":
    recommender = BanglaRecommender(ARTICLE_DIR, STOPWORDS_FILE)

    while True:
        query = input("\n🔍 Enter your Bangla search query (or 'exit'): ")
        if query.lower() == "exit":
            break
        recommendations = recommender.recommend(query)
        print("\nTop Matches:")
        for name, score in recommendations:
            print(f"{name}: similarity={score:.4f}")
