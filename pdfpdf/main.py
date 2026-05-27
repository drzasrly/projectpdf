from loader import DocumentLoader
from preprocessing import preprocess_text
from feature_extraction import FeatureExtraction

# =========================
# EUCLIDEAN DISTANCE
# =========================
def euclidean_distance(vec1, vec2):
    total = 0
    for key in vec1:
        total += (vec1[key] - vec2[key]) ** 2
    return total ** 0.5


# =========================
# PRINT VECTOR QUERY
# =========================
def print_query_vector(vector, vocab):
    print("\nVektor")
    header = "Vektor".ljust(10)
    for word in vocab:
        header += word.ljust(10)
    header += "Skor"
    print(header)
    print("-"*120)

    row = "QUERY".ljust(10)
    for word in vocab:
        row += f"{vector[word]:.4f}".ljust(10)
    row += "-"
    print(row)


def main():
    # =========================
    # LOAD DATA
    # =========================
    loader = DocumentLoader("dataset")
    docs = loader.get_as_dict()

    print(type(docs))
    print("Jumlah dokumen:", len(docs))

    # =========================
    # CONTOH DOKUMEN
    # =========================
    doc_id = list(docs.keys())[0]
    print("Contoh dokumen:", doc_id)
    print(docs[doc_id])
    print("=" * 50)

    # =========================
    # PREPROCESSING DETAIL
    # =========================
    processed_docs = []

    for doc_id, text in docs.items():

        print("Dokumen:", doc_id)
        print("-" * 40)

        hasil = preprocess_text(text["content"])

        print("Case Folding:")
        print(hasil["case_folding"])

        print("\nTokens:")
        print(hasil["tokens"])

        print("\nAfter Stopword Removal:")
        print(hasil["stopword_removed"])

        print("\nAfter Stemming:")
        print(hasil["stemming"])

        print("=" * 50)

        processed_docs.append(hasil["stemming"])

    # =========================
    # FEATURE EXTRACTION
    # =========================
    fe = FeatureExtraction(processed_docs)

    vocab = fe.vocab
    bow = fe.compute_bow()
    tfidf = fe.compute_tfidf()
    idf = fe.compute_idf()

    # =========================
    # PRINT BoW
    # =========================
    print("\n" + "="*70)
    print("BAG OF WORDS (BoW)")
    print("="*70)

    header = "Term".ljust(15)
    for i in range(len(bow)):
        header += f"D{i+1}".ljust(10)
    print(header)
    print("-"*70)

    for word in vocab:
        row = word.ljust(15)
        for doc in bow:
            row += str(doc[word]).ljust(10)
        print(row)

    # =========================
    # PRINT TF-IDF
    # =========================
    print("\n" + "="*70)
    print("TF-IDF")
    print("="*70)

    header = "Term".ljust(15)
    for i in range(len(tfidf)):
        header += f"D{i+1}".ljust(12)
    print(header)
    print("-"*70)

    for word in vocab:
        row = word.ljust(15)
        for doc in tfidf:
            row += f"{doc[word]:.4f}".ljust(12)
        print(row)

    # =========================
    # PRINT IDF
    # =========================
    print("\n" + "="*70)
    print("IDF")
    print("="*70)

    for word, val in idf.items():
        print(f"{word.ljust(15)} : {val:.4f}")

    print("\nVocabulary:", vocab)

    # =========================
    # QUERY & EUCLIDEAN DISTANCE (STYLE TABEL)
    # =========================
    print("\n" + "="*120)
    query = input("Masukkan query: ")

    print("\nANALISIS QUERY:", query)
    print("MENGGUNAKAN METRIK: Euclidean Distance")
    print("="*120)

    hasil_query = preprocess_text(query)
    query_tokens = hasil_query["stemming"]

    print("\nTokens Query (Stemmed):", query_tokens)

    # TF QUERY
    query_tf = dict.fromkeys(vocab, 0)
    for word in query_tokens:
        if word in query_tf:
            query_tf[word] += 1

    # TF-IDF QUERY
    query_tfidf = {}
    for word in vocab:
        query_tfidf[word] = query_tf[word] * idf[word]

    # PRINT VECTOR QUERY
    print_query_vector(query_tfidf, vocab)

    # =========================
    # HITUNG JARAK & RANKING
    # =========================
    print("\nRanking Dokumen (Berdasarkan Euclidean Distance):")

    results = []

    for i, doc_vec in enumerate(tfidf):
        dist = euclidean_distance(query_tfidf, doc_vec)
        results.append((f"doc{i+1}.txt", doc_vec, dist))

    # sorting
    results = sorted(results, key=lambda x: x[2])

    # tampilkan
    for rank, (doc_name, vec, dist) in enumerate(results, start=1):
        row = doc_name.ljust(12)

        for word in vocab:
            row += f"{vec[word]:.4f}".ljust(10)

        row += f"{dist:.4f} (Rank {rank})"
        print(row)


if __name__ == "__main__":
    main()