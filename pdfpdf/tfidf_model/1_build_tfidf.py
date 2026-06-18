import sys
import os
import csv

# Tambahkan path parent directory agar bisa mengimpor module asli (loader, preprocessing, feature_extraction)
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from loader import DocumentLoader
from preprocessing import preprocess_text
from feature_extraction import FeatureExtraction

def main():
    print("===================================")
    print("MEMBANGUN MODEL TF-IDF")
    print("===================================")

    # Load documents dari folder dataset
    dataset_path = os.path.join(parent_dir, "dataset")
    loader = DocumentLoader(dataset_path)
    docs = loader.get_as_dict()

    if not docs:
        print("Tidak ada dokumen yang dimuat.")
        return

    # Preprocessing
    processed_docs = []
    doc_filenames = list(docs.keys())

    for doc_id, text in docs.items():
        hasil = preprocess_text(text["content"])
        processed_docs.append(hasil["stemming"])

    # Feature Extraction
    fe = FeatureExtraction(processed_docs)
    vocab = fe.vocab
    tfidf = fe.compute_tfidf()
    idf = fe.compute_idf()

    # Simpan VOCABULARY ke vocab.csv
    with open("vocab.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Index", "Word"])
        for i, word in enumerate(vocab):
            writer.writerow([i, word])
            
    # Simpan IDF ke idf.csv
    with open("idf.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Word", "IDF_Score"])
        for word, score in idf.items():
            writer.writerow([word, score])

    # Simpan TF-IDF Vectors (Matrix Doc x Term)
    with open("tfidf_vectors.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["Filename"] + vocab
        writer.writerow(header)
        
        for i, doc_vector in enumerate(tfidf):
            row = [doc_filenames[i]]
            for word in vocab:
                row.append(doc_vector[word])
            writer.writerow(row)

    print("\n✅ Berhasil mengekstrak dan menyimpan model TF-IDF:")
    print("1. vocab.csv")
    print("2. idf.csv")
    print("3. tfidf_vectors.csv")

if __name__ == "__main__":
    main()
