import sys
import os
import csv
import math

# Tambahkan path parent directory untuk bisa mengimpor preprocessing.py
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from preprocessing import preprocess_text

def euclidean_distance(vec1, vec2, vocab):
    total = 0
    for word in vocab:
        total += (vec1.get(word, 0) - vec2.get(word, 0)) ** 2
    return total ** 0.5

def cosine_similarity(vec1, vec2, vocab):
    dot_product = 0
    norm_a = 0
    norm_b = 0
    for word in vocab:
        val1 = vec1.get(word, 0)
        val2 = vec2.get(word, 0)
        dot_product += val1 * val2
        norm_a += val1 ** 2
        norm_b += val2 ** 2
        
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (math.sqrt(norm_a) * math.sqrt(norm_b))

def main():
    print("===================================")
    print("MEMUAT MODEL TF-IDF TERSIMPAN")
    print("===================================")

    if not os.path.exists("vocab.csv") or not os.path.exists("idf.csv") or not os.path.exists("tfidf_vectors.csv"):
        print("Model belum di-build. Silakan jalankan 'python 1_build_tfidf.py' terlebih dahulu.")
        return

    # Load Vocab
    vocab = []
    with open("vocab.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            vocab.append(row[1])

    # Load IDF
    idf = {}
    with open("idf.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            idf[row[0]] = float(row[1])

    # Load TF-IDF Vectors
    docs_tfidf = []
    with open("tfidf_vectors.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        
        for row in reader:
            filename = row[0]
            vector = {}
            for i, word in enumerate(vocab):
                vector[word] = float(row[i+1])
            docs_tfidf.append({"filename": filename, "vector": vector})

    print(f"Berhasil memuat {len(vocab)} kata vocabulary dan vektor TF-IDF untuk {len(docs_tfidf)} dokumen.")
    print("Ketik 'exit' atau 'keluar' untuk berhenti.")

    while True:
        print("\n" + "="*80)
        query = input("Masukkan query pencarian dokumen: ").strip()

        if query.lower() in ['exit', 'keluar', 'quit']:
            print("Program dihentikan.")
            break

        # Preprocess query menggunakan modul asli
        hasil_query = preprocess_text(query)
        query_tokens = hasil_query["stemming"]
        
        if not query_tokens:
            print("Query kosong setelah preprocessing (mungkin hanya berisi angka, simbol, atau stopword).")
            continue

        print("\nTokens Query (Setelah Preprocessing):", query_tokens)

        # Hitung TF Query
        query_tf = dict.fromkeys(vocab, 0)
        for word in query_tokens:
            if word in query_tf:
                query_tf[word] += 1

        # Hitung TF-IDF Query
        query_tfidf = {}
        for word in vocab:
            query_tfidf[word] = query_tf[word] * idf[word]

        # Hitung Jarak (Similarity) terhadap dokumen-dokumen
        results = []
        for doc in docs_tfidf:
            dist_euclid = euclidean_distance(query_tfidf, doc["vector"], vocab)
            sim_cosine = cosine_similarity(query_tfidf, doc["vector"], vocab)
            results.append({
                "filename": doc["filename"],
                "euclidean": dist_euclid,
                "cosine": sim_cosine
            })

        # Urutkan berdasarkan jarak terkecil (Euclidean) - yang paling mirip
        results = sorted(results, key=lambda x: x["euclidean"])

        print("\nRanking Dokumen (Berdasarkan Euclidean Distance Terkecil):")
        print(f"{'Rank':<5} | {'Dokumen':<60} | {'Euclidean Dist':<20} | {'Cosine Sim':<20}")
        print("-" * 115)
        
        for rank, res in enumerate(results, 1):
            print(f"{rank:<5} | {res['filename']:<60} | {res['euclidean']:<20.4f} | {res['cosine']:.4f}")

if __name__ == "__main__":
    main()
