import os
import re
import csv
import numpy as np
from PyPDF2 import PdfReader

# =====================================================
# NLTK
# =====================================================
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# download stopwords pertama kali (optional, assume already downloaded)
# nltk.download('stopwords')

# =====================================================
# PARAMETER
# =====================================================
DATASET_FOLDER = "../dataset"
MAX_KATA = 50
WINDOW_SIZE = 2
EMBEDDING_DIM = 5

# =====================================================
# STEMMER & STOPWORDS
# =====================================================
stemmer = PorterStemmer()

try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# =====================================================
# PREPROCESSING FUNCTION
# =====================================================
def preprocess_text(text):
    # lowercase
    text = text.lower()
    
    # hapus angka & simbol
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # tokenisasi
    tokens = text.split()
    
    # hapus stopwords
    tokens = [word for word in tokens if word not in stop_words]
    
    # stemming
    tokens = [stemmer.stem(word) for word in tokens]
    
    # maksimal kata
    tokens = tokens[:MAX_KATA]
    
    return tokens

# =====================================================
# MEMBACA SEMUA FILE PDF
# =====================================================
documents = []

# Resolve paths
base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(base_dir, DATASET_FOLDER)

pdf_files = [
    file for file in os.listdir(dataset_path)
    if file.endswith(".pdf")
]

print("\n===================================")
print("MEMBACA FILE PDF")
print("===================================")

for pdf_file in pdf_files:
    path = os.path.join(dataset_path, pdf_file)
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        try:
            extracted = page.extract_text()
            if extracted:
                text += extracted + " "
        except Exception as e:
            pass

    documents.append({
        "filename": pdf_file,
        "text": text
    })
    print(f"Loaded : {pdf_file}")

# =====================================================
# PREPROCESSING
# =====================================================
processed_docs = []

print("\n===================================")
print("PREPROCESSING")
print("===================================")

for doc in documents:
    tokens = preprocess_text(doc["text"])
    processed_docs.append(tokens)
    print(f"\n{doc['filename']}")
    print(tokens[:30])

# =====================================================
# SIMPAN HASIL PREPROCESSING KE CSV
# =====================================================
with open("storage.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["No", "Filename", "Preprocessing"])
    for i, doc in enumerate(processed_docs):
        writer.writerow([i + 1, documents[i]["filename"], ' '.join(doc)])

print("\n===================================")
print("storage.csv BERHASIL DISIMPAN")
print("===================================")

# =====================================================
# MEMBUAT VOCABULARY
# =====================================================
vocab = set()

for doc in processed_docs:
    for word in doc:
        vocab.add(word)

vocab = sorted(vocab)

# =====================================================
# WORD INDEX
# =====================================================
word_to_index = {word: i for i, word in enumerate(vocab)}
index_to_word = {i: word for word, i in word_to_index.items()}
vocab_size = len(vocab)

# =====================================================
# SIMPAN KAMUS KE CSV
# =====================================================
with open("kamus.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Index", "Word"])
    for word, idx in word_to_index.items():
        writer.writerow([idx, word])

print("\n===================================")
print("kamus.csv BERHASIL DISIMPAN")
print("===================================")

# =====================================================
# TAMPILKAN VOCABULARY
# =====================================================
print("\n===================================")
print("VOCABULARY")
print("===================================")
for word, idx in word_to_index.items():
    print(f"{idx} : {word}")

# =====================================================
# MEMBENTUK X_train DAN y_train (CBOW)
# =====================================================
X_train = []
y_train = []
X_train_words = []
y_train_words = []

for tokens in processed_docs:
    for i, target_word in enumerate(tokens):
        start = max(0, i - WINDOW_SIZE)
        end = min(len(tokens), i + WINDOW_SIZE + 1)
        
        context_indices = []
        context_words = []
        
        for j in range(start, end):
            if i != j:
                context_word = tokens[j]
                context_indices.append(word_to_index[context_word])
                context_words.append(context_word)
        
        if len(context_indices) > 0:
            X_train.append(context_indices)
            X_train_words.append(context_words)
            y_train.append(word_to_index[target_word])
            y_train_words.append(target_word)

# =====================================================
# TAMPILKAN X_train DAN y_train
# =====================================================
print("\n===================================")
print("X_train (Context) DAN y_train (Target) - CBOW")
print("===================================")
for i in range(len(X_train)):
    print(
        f"Data-{i+1}"
        f" | X_train : {X_train[i]} ({X_train_words[i]})"
        f" --> "
        f"y_train : {y_train[i]} ({y_train_words[i]})"
    )

# =====================================================
# SIMPAN X_train DAN y_train KE CSV
# =====================================================
with open("x_y_train_cbow.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        "No",
        "X_train_Context_Indices",
        "X_train_Context_Words",
        "y_train_Target_Index",
        "y_train_Target_Word"
    ])
    for i in range(len(X_train)):
        writer.writerow([
            i + 1,
            str(X_train[i]),
            str(X_train_words[i]),
            y_train[i],
            y_train_words[i]
        ])

print("\n===================================")
print("x_y_train_cbow.csv BERHASIL DISIMPAN")
print("===================================")

# =====================================================
# ONE HOT ENCODING
# =====================================================
def one_hot(index, vocab_size):
    vector = np.zeros(vocab_size)
    vector[index] = 1
    return vector

print("\n===================================")
print("ONE HOT ENCODING (CBOW)")
print("===================================")

for i in range(min(20, len(X_train))):
    print("\n-----------------------------------")
    print(f"Context : {X_train_words[i]}")
    
    # Untuk CBOW biasanya One Hot vector dari Context dijumlahkan atau dirata-rata
    context_vector = np.zeros(vocab_size)
    for ctx_idx in X_train[i]:
        context_vector += one_hot(ctx_idx, vocab_size)
        
    context_vector = context_vector / len(X_train[i])
    
    print(np.round(context_vector, 2))
    
    print(f"Target : {y_train_words[i]}")
    y_vector = one_hot(y_train[i], vocab_size)
    print(y_vector.astype(int))

# =====================================================
# SIMPAN ONE HOT KE CSV
# =====================================================
with open("onehot_encoding.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    header = ["Word"]
    for i in range(vocab_size):
        header.append(f"V{i}")
    writer.writerow(header)
    for word, idx in word_to_index.items():
        vector = one_hot(idx, vocab_size).astype(int)
        row = [word]
        row.extend(vector)
        writer.writerow(row)

print("\n===================================")
print("onehot_encoding.csv BERHASIL DISIMPAN")
print("===================================")

# =====================================================
# MATRIX BOBOT DIMENSI (MBD)
# =====================================================
MBD = np.random.rand(vocab_size, EMBEDDING_DIM)

print("\n===================================")
print("MATRIX BOBOT DIMENSI (MBD)")
print("===================================")
for i in range(vocab_size):
    print(f"{index_to_word[i]} : {MBD[i]}")

# =====================================================
# SIMPAN MBD KE CSV
# =====================================================
with open("mbd.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    header = ["Word"]
    for i in range(EMBEDDING_DIM):
        header.append(f"Dim-{i+1}")
    writer.writerow(header)
    for i in range(vocab_size):
        row = [index_to_word[i]]
        row.extend(MBD[i])
        writer.writerow(row)

print("\n===================================")
print("mbd.csv BERHASIL DISIMPAN")
print("===================================")

# =====================================================
# INFO DATA
# =====================================================
print("\n===================================")
print("INFO DATA")
print("===================================")
print("Jumlah PDF :", len(documents))
print("Jumlah Vocabulary :", vocab_size)
print("Jumlah Pasangan CBOW (X_train) :", len(X_train))
print("Shape MBD :", MBD.shape)
