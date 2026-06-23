import os
import csv
import math
import numpy as np

from loader import DocumentLoader
from preprocessing import preprocess_text

# =====================================================
# KONFIGURASI GROUND TRUTH (DATA UJI)
# =====================================================
TEST_QUERIES = []
if os.path.exists("ground_truth.csv"):
    with open("ground_truth.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        queries_map = {}
        for row in reader:
            q = row.get("query", "").strip()
            doc = row.get("document", "").strip()
            if q and doc:
                if q not in queries_map:
                    queries_map[q] = []
                queries_map[q].append(doc)
        
        for q, docs in queries_map.items():
            TEST_QUERIES.append({
                "query": q,
                "relevant_docs": docs
            })
else:
    print("WARNING: file ground_truth.csv tidak ditemukan. Evaluasi mungkin tidak berjalan dengan baik.")
    TEST_QUERIES = []

# =====================================================
# LOADERS & UTILITIES
# =====================================================
def load_tfidf_model():
    vocab, idf, tfidf_docs = [], {}, []
    
    if not os.path.exists("tfidf_model/vocab.csv"): return None
    with open("tfidf_model/vocab.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader: vocab.append(row[1])

    with open("tfidf_model/idf.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader: idf[row[0]] = float(row[1])

    with open("tfidf_model/tfidf_vectors.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            filename = row[0]
            vector = {vocab[i]: float(row[i+1]) for i in range(len(vocab))}
            tfidf_docs.append({"filename": filename, "vector": vector})
            
    return vocab, idf, tfidf_docs

def load_w2v_model(folder):
    if not os.path.exists(f"{folder}/kamus.csv"): return None
    
    word_to_index = {}
    with open(f"{folder}/kamus.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader: word_to_index[row[1]] = int(row[0])
        
    mbd_vectors = {}
    with open(f"{folder}/mbd.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            mbd_vectors[row[0]] = np.array([float(x) for x in row[1:]])
            
    return word_to_index, mbd_vectors

# =====================================================
# SIMILARITY & DISTANCE
# =====================================================
def cosine_similarity_dict(vec1, vec2, vocab):
    dot_product = sum(vec1.get(w, 0) * vec2.get(w, 0) for w in vocab)
    norm_a = sum(v**2 for v in vec1.values())
    norm_b = sum(v**2 for v in vec2.values())
    if norm_a == 0 or norm_b == 0: return 0.0
    return dot_product / (math.sqrt(norm_a) * math.sqrt(norm_b))

def euclidean_distance_dict(vec1, vec2, vocab):
    return math.sqrt(sum((vec1.get(w, 0) - vec2.get(w, 0)) ** 2 for w in vocab))

def cosine_similarity_arr(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norm_a, norm_b = np.linalg.norm(vec1), np.linalg.norm(vec2)
    if norm_a == 0 or norm_b == 0: return 0.0
    return dot / (norm_a * norm_b)

def euclidean_distance_arr(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

def word_by_word_similarity(q_tokens, d_tokens, mbd_vectors, metric="cosine"):
    q_vecs = [mbd_vectors[w] for w in q_tokens if w in mbd_vectors]
    d_vecs = [mbd_vectors[w] for w in d_tokens if w in mbd_vectors]
    
    if not q_vecs or not d_vecs:
        return 0.0 if metric == "cosine" else 9999.0
        
    total_sim = 0.0
    count = 0
    for qv in q_vecs:
        for dv in d_vecs:
            if metric == "cosine":
                total_sim += cosine_similarity_arr(qv, dv)
            else:
                total_sim += euclidean_distance_arr(qv, dv)
            count += 1
            
    return total_sim / count if count > 0 else (0.0 if metric == "cosine" else 9999.0)

# =====================================================
# EVALUATION METRICS (Top-K)
# =====================================================
def calculate_metrics(ranked_docs, relevant_docs, k=5):
    top_k = ranked_docs[:k]
    hits = sum(1 for d in top_k if d in relevant_docs)
    
    precision = hits / k if k > 0 else 0
    recall = hits / len(relevant_docs) if len(relevant_docs) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    sum_precs = 0
    h = 0
    for i, doc in enumerate(top_k):
        if doc in relevant_docs:
            h += 1
            sum_precs += h / (i + 1.0)
    ap = sum_precs / min(len(relevant_docs), k) if len(relevant_docs) > 0 else 0
    
    mrr = 0
    for i, doc in enumerate(top_k):
        if doc in relevant_docs:
            mrr = 1.0 / (i + 1.0)
            break
            
    dcg = sum(1.0 / math.log2(i + 2) for i, doc in enumerate(top_k) if doc in relevant_docs)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant_docs), k)))
    ndcg = dcg / idcg if idcg > 0 else 0
    
    return precision, recall, f1, ap, mrr, ndcg

# =====================================================
# MAIN
# =====================================================
def main():
    print("Memuat dokumen dan melakukan pre-processing untuk kalkulasi...")
    loader = DocumentLoader("dataset")
    docs = loader.get_as_dict()
    
    doc_tokens = {}
    for filename, text in docs.items():
        hasil = preprocess_text(text["content"])
        doc_tokens[filename] = hasil["stemming"]
        
    print("Memuat Model...")
    tfidf_data = load_tfidf_model()
    if not tfidf_data:
        print("TF-IDF model belum di-build.")
        return
    tfidf_vocab, tfidf_idf, tfidf_docs = tfidf_data
    
    cbow_data = load_w2v_model("word2vec_cbow")
    if not cbow_data:
        print("CBOW model belum di-build.")
        return
    cbow_w2i, cbow_mbd = cbow_data
    
    skipgram_data = load_w2v_model(".")
    if not skipgram_data:
        print("Skip-Gram model belum di-build.")
        return
    sg_w2i, sg_mbd = skipgram_data

    # ==========================
    # EVALUATION
    # ==========================
    results_agg = {"TF-IDF": [], "CBOW": [], "Skip-Gram": []}
    
    for item in TEST_QUERIES:
        query = item["query"]
        relevant = item["relevant_docs"]
        
        q_tokens = preprocess_text(query)["stemming"]
        
        q_tf = dict.fromkeys(tfidf_vocab, 0)
        for w in q_tokens: 
            if w in q_tf: q_tf[w] += 1
        q_tfidf_vec = {w: q_tf[w] * tfidf_idf[w] for w in tfidf_vocab}
        
        ranks_tfidf = []
        for d in tfidf_docs:
            sim = cosine_similarity_dict(q_tfidf_vec, d["vector"], tfidf_vocab)
            ranks_tfidf.append((d["filename"], sim))
        ranks_tfidf.sort(key=lambda x: x[1], reverse=True)
        ranked_files_tfidf = [x[0] for x in ranks_tfidf]
        
        ranks_cbow = []
        for f, d_tokens in doc_tokens.items():
            sim = word_by_word_similarity(q_tokens, d_tokens, cbow_mbd, metric="cosine")
            ranks_cbow.append((f, sim))
        ranks_cbow.sort(key=lambda x: x[1], reverse=True)
        ranked_files_cbow = [x[0] for x in ranks_cbow]
        
        ranks_sg = []
        for f, d_tokens in doc_tokens.items():
            sim = word_by_word_similarity(q_tokens, d_tokens, sg_mbd, metric="cosine")
            ranks_sg.append((f, sim))
        ranks_sg.sort(key=lambda x: x[1], reverse=True)
        ranked_files_sg = [x[0] for x in ranks_sg]
        
        results_agg["TF-IDF"].append(calculate_metrics(ranked_files_tfidf, relevant, 5))
        results_agg["CBOW"].append(calculate_metrics(ranked_files_cbow, relevant, 5))
        results_agg["Skip-Gram"].append(calculate_metrics(ranked_files_sg, relevant, 5))
        
    print("\n" + "="*100)
    print("HASIL RATA-RATA EVALUASI")
    print("="*100)
    print(f"{'Model':<15} {'Precision':<10} {'Recall':<10} {'F1':<10} {'MAP':<10} {'MRR':<10} {'NDCG':<10}")
    print("-" * 100)
    
    eval_csv_data = [["Model", "Precision", "Recall", "F1", "MAP", "MRR", "NDCG"]]
    for model in ["TF-IDF", "CBOW", "Skip-Gram"]:
        if len(results_agg[model]) > 0:
            avgs = np.mean(results_agg[model], axis=0)
        else:
            avgs = [0.0] * 6
        print(f"{model:<15} {avgs[0]:<10.4f} {avgs[1]:<10.4f} {avgs[2]:<10.4f} {avgs[3]:<10.4f} {avgs[4]:<10.4f} {avgs[5]:<10.4f}")
        eval_csv_data.append([model, f"{avgs[0]:.4f}", f"{avgs[1]:.4f}", f"{avgs[2]:.4f}", f"{avgs[3]:.4f}", f"{avgs[4]:.4f}", f"{avgs[5]:.4f}"])
        
    print("="*100)
    
    with open("hasil_evaluasi.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(eval_csv_data)
    print("\n[SYSTEM] hasil_evaluasi.csv berhasil disimpan")

    # ==========================
    # INTERACTIVE SEARCH
    # ==========================
    print("\n[SYSTEM] Memasuki Mode Pencarian Interaktif")
    while True:
        print("\n" + "="*80)
        query = input("Masukkan query pencarian (ketik 'exit' untuk keluar): ").strip()
        if query.lower() in ['exit', 'keluar', 'quit']: break
        
        q_tokens = preprocess_text(query)["stemming"]
        if not q_tokens:
            print("Query kosong setelah preprocessing.")
            continue
            
        print(f"\nTokens Query: {q_tokens}")
        
        # --- TF-IDF ---
        q_tf = dict.fromkeys(tfidf_vocab, 0)
        for w in q_tokens: 
            if w in q_tf: q_tf[w] += 1
        q_tfidf_vec = {w: q_tf[w] * tfidf_idf[w] for w in tfidf_vocab}
        
        res_tfidf_docs = []
        for d in tfidf_docs:
            cos = cosine_similarity_dict(q_tfidf_vec, d["vector"], tfidf_vocab)
            euc = euclidean_distance_dict(q_tfidf_vec, d["vector"], tfidf_vocab)
            res_tfidf_docs.append((d["filename"], cos, euc))
        res_tfidf_docs.sort(key=lambda x: x[1], reverse=True)
        
        # --- CBOW ---
        res_cbow_docs = []
        for f, d_tokens in doc_tokens.items():
            cos = word_by_word_similarity(q_tokens, d_tokens, cbow_mbd, metric="cosine")
            euc = word_by_word_similarity(q_tokens, d_tokens, cbow_mbd, metric="euclidean")
            res_cbow_docs.append((f, cos, euc))
        res_cbow_docs.sort(key=lambda x: x[1], reverse=True)
        
        res_cbow_words = []
        for w, v in cbow_mbd.items():
            if w not in q_tokens:
                cos = word_by_word_similarity(q_tokens, [w], cbow_mbd, metric="cosine")
                euc = word_by_word_similarity(q_tokens, [w], cbow_mbd, metric="euclidean")
                res_cbow_words.append((w, cos, euc))
        res_cbow_words.sort(key=lambda x: x[1], reverse=True)
        
        # --- Skip-Gram ---
        res_sg_docs = []
        for f, d_tokens in doc_tokens.items():
            cos = word_by_word_similarity(q_tokens, d_tokens, sg_mbd, metric="cosine")
            euc = word_by_word_similarity(q_tokens, d_tokens, sg_mbd, metric="euclidean")
            res_sg_docs.append((f, cos, euc))
        res_sg_docs.sort(key=lambda x: x[1], reverse=True)
        
        res_sg_words = []
        for w, v in sg_mbd.items():
            if w not in q_tokens:
                cos = word_by_word_similarity(q_tokens, [w], sg_mbd, metric="cosine")
                euc = word_by_word_similarity(q_tokens, [w], sg_mbd, metric="euclidean")
                res_sg_words.append((w, cos, euc))
        res_sg_words.sort(key=lambda x: x[1], reverse=True)
        
        # Print Results
        print("\n>>> HASIL PENCARIAN (Top 3 Dokumen & Kata) <<<")
        
        # 1. TF-IDF
        print(f"\n[MODEL: TF-IDF]")
        print("=> Top 3 Dokumen Terkait:")
        print(f"{'No':<3} | {'Cosine Sim':<12} | {'Euclidean':<12} | {'Nama Dokumen'}")
        print("-" * 100)
        for i in range(3):
            if i < len(res_tfidf_docs):
                print(f"{i+1:<3} | {res_tfidf_docs[i][1]:<12.4f} | {res_tfidf_docs[i][2]:<12.4f} | {res_tfidf_docs[i][0]}")
                
        # 2. CBOW
        print(f"\n[MODEL: CBOW]")
        print("=> Top 3 Dokumen Terkait:")
        print(f"{'No':<3} | {'Cosine Sim':<12} | {'Euclidean':<12} | {'Nama Dokumen'}")
        print("-" * 100)
        for i in range(3):
            if i < len(res_cbow_docs):
                print(f"{i+1:<3} | {res_cbow_docs[i][1]:<12.4f} | {res_cbow_docs[i][2]:<12.4f} | {res_cbow_docs[i][0]}")
        print("=> Top 3 Kata Paling Mirip:")
        print(f"{'No':<3} | {'Cosine Sim':<12} | {'Euclidean':<12} | {'Kata'}")
        print("-" * 50)
        for i in range(3):
            if i < len(res_cbow_words):
                print(f"{i+1:<3} | {res_cbow_words[i][1]:<12.4f} | {res_cbow_words[i][2]:<12.4f} | {res_cbow_words[i][0]}")
                
        # 3. Skip-Gram
        print(f"\n[MODEL: Skip-Gram]")
        print("=> Top 3 Dokumen Terkait:")
        print(f"{'No':<3} | {'Cosine Sim':<12} | {'Euclidean':<12} | {'Nama Dokumen'}")
        print("-" * 100)
        for i in range(3):
            if i < len(res_sg_docs):
                print(f"{i+1:<3} | {res_sg_docs[i][1]:<12.4f} | {res_sg_docs[i][2]:<12.4f} | {res_sg_docs[i][0]}")
        print("=> Top 3 Kata Paling Mirip:")
        print(f"{'No':<3} | {'Cosine Sim':<12} | {'Euclidean':<12} | {'Kata'}")
        print("-" * 50)
        for i in range(3):
            if i < len(res_sg_words):
                print(f"{i+1:<3} | {res_sg_words[i][1]:<12.4f} | {res_sg_words[i][2]:<12.4f} | {res_sg_words[i][0]}")

if __name__ == "__main__":
    main()
