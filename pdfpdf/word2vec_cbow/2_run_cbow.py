import csv
import numpy as np
import os
import math

# =====================================================
# LOAD KAMUS (VOCABULARY)
# =====================================================
def load_kamus(filepath):
    word_to_index = {}
    index_to_word = {}
    
    if not os.path.exists(filepath):
        print(f"File {filepath} tidak ditemukan. Harap jalankan script build terlebih dahulu.")
        return None, None

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader) # skip header
        for row in reader:
            idx = int(row[0])
            word = row[1]
            word_to_index[word] = idx
            index_to_word[idx] = word
            
    return word_to_index, index_to_word

# =====================================================
# LOAD MBD (EMBEDDING WEIGHTS)
# =====================================================
def load_mbd(filepath, vocab_size):
    if not os.path.exists(filepath):
        print(f"File {filepath} tidak ditemukan.")
        return None
        
    mbd_vectors = {}
    
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader) # skip header
        for row in reader:
            word = row[0]
            # Convert str dimensions back to float
            vector = np.array([float(x) for x in row[1:]])
            mbd_vectors[word] = vector
            
    return mbd_vectors

# =====================================================
# COSINE SIMILARITY
# =====================================================
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

# =====================================================
# EUCLIDEAN DISTANCE
# =====================================================
def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

# =====================================================
# MAIN RUN LOOP
# =====================================================
def main():
    print("\n===================================")
    print("MEMUAT MODEL WORD2VEC CBOW")
    print("===================================")
    
    word_to_index, index_to_word = load_kamus("kamus.csv")
    
    if word_to_index is None:
        return
        
    vocab_size = len(word_to_index)
    mbd_vectors = load_mbd("mbd.csv", vocab_size)
    
    if mbd_vectors is None:
        return
        
    print(f"Berhasil memuat {vocab_size} kata beserta bobot dimensinya (MBD).")
    print("Model siap digunakan!")
    print("Ketik 'exit' atau 'keluar' untuk mengakhiri program.")
    
    while True:
        print("\n" + "="*50)
        query = input("Masukkan satu kata query: ").strip().lower()
        
        if query in ['exit', 'keluar', 'quit']:
            print("Program dihentikan.")
            break
            
        if query not in word_to_index:
            print(f"Kata '{query}' tidak ditemukan di dalam vocabulary (Kamus).")
            continue
            
        query_vector = mbd_vectors[query]
        print(f"\nKata: '{query}'")
        print(f"Vektor Embedding: {query_vector}")
        
        # Hitung kedekatan dengan semua kata di vocab
        similarities = []
        for word, vector in mbd_vectors.items():
            if word != query:
                cos_sim = cosine_similarity(query_vector, vector)
                euclid_dist = euclidean_distance(query_vector, vector)
                similarities.append({
                    "word": word,
                    "cosine": cos_sim,
                    "euclidean": euclid_dist
                })
                
        # Urutkan berdasarkan Cosine Similarity tertinggi
        similarities = sorted(similarities, key=lambda x: x["cosine"], reverse=True)
        
        print(f"\n5 Kata Paling Mirip dengan '{query}':")
        print(f"{'No':<5} | {'Kata':<20} | {'Cosine Sim (Mendekati 1)':<25} | {'Euclidean Dist (Mendekati 0)'}")
        print("-" * 85)
        
        for i in range(5):
            if i < len(similarities):
                data = similarities[i]
                print(f"{i+1:<5} | {data['word']:<20} | {data['cosine']:<25.4f} | {data['euclidean']:.4f}")

if __name__ == "__main__":
    main()
