from flask import Flask, render_template, request, send_from_directory
import json
import os
from preprocessing import preprocess_text
from feature_extraction import FeatureExtraction

app = Flask(__name__)
base_dir = os.path.dirname(os.path.abspath(__file__))

# 🔥 COSINE SIMILARITY
def cosine_similarity(vec1, vec2):
    dot = sum(vec1[k] * vec2[k] for k in vec1)
    norm1 = sum(vec1[k]**2 for k in vec1) ** 0.5
    norm2 = sum(vec2[k]**2 for k in vec2) ** 0.5
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot / (norm1 * norm2)

# LOAD JSON
with open(os.path.join(base_dir, "data.json"), "r", encoding="utf-8") as f:
    raw_docs = json.load(f)

doc_names = []
processed_docs = []

# 🔥 HANYA ABSTRACT
for name, data in raw_docs.items():
    hasil = preprocess_text(data["abstract"])
    processed_docs.append(hasil["stemming"])
    doc_names.append(name)

# TF-IDF
fe = FeatureExtraction(processed_docs)
vocab = fe.vocab
tfidf_docs = fe.compute_tfidf()
idf = fe.compute_idf()

# ROUTE PDF
@app.route("/pdf/<filename>")
def pdf_view(filename):
    return send_from_directory(os.path.join(base_dir, "dataset"), filename)

# 🔍 SEARCH
@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    query = ""

    if request.method == "POST":
        query = request.form["query"]

        hasil = preprocess_text(query)
        tokens = hasil["stemming"]

        query_tf = dict.fromkeys(vocab, 0)
        for word in tokens:
            if word in query_tf:
                query_tf[word] += 1

        query_vec = {w: query_tf[w]*idf[w] for w in vocab}

        threshold = 0.1  # 🔥 FILTER

        for i, doc_vec in enumerate(tfidf_docs):
            score = cosine_similarity(query_vec, doc_vec)

            if score >= threshold:
                results.append({
                    "name": doc_names[i],
                    "title": raw_docs[doc_names[i]]["title"],
                    "authors": raw_docs[doc_names[i]]["authors"],
                    "year": raw_docs[doc_names[i]]["year"],
                    "abstract": raw_docs[doc_names[i]]["abstract"][:200],
                    "score": round(score, 4)
                })

        results = sorted(results, key=lambda x: x["score"], reverse=True)

    return render_template("index.html", results=results, query=query)

if __name__ == "__main__":
    app.run(debug=True)