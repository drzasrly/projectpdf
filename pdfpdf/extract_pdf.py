import os
import json
import pdfplumber
import re

DATASET_PATH = "dataset"
OUTPUT_FILE = "data.json"

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def extract_title(text):
    lines = text.split("\n")
    for line in lines[:15]:
        line = line.strip()
        if len(line) < 10:
            continue
        if any(x in line.lower() for x in ["journal", "volume", "doi"]):
            continue
        if re.search(r'\d{4}', line):
            continue
        return line
    return "No Title"

def extract_abstract(text):
    text_lower = text.lower()

    # cari posisi "abstract"
    start = text_lower.find("abstract")
    if start == -1:
        return "Abstract tidak ditemukan"

    # ambil setelah kata abstract
    abstract_text = text[start + len("abstract"):]

    # batas akhir (biasanya sebelum introduction / keywords)
    end_keywords = ["introduction", "keywords", "background"]

    end_pos = len(abstract_text)
    for word in end_keywords:
        pos = abstract_text.lower().find(word)
        if pos != -1:
            end_pos = min(end_pos, pos)

    return abstract_text[:end_pos].strip()

def extract_year(text):
    match = re.search(r'(19|20)\d{2}', text)
    return match.group(0) if match else "Unknown"

def extract_authors(text):
    lines = text.split("\n")
    for line in lines[:20]:
        if "," in line:
            return line.strip()
    return "Unknown"

data = {}

for file in os.listdir(DATASET_PATH):
    if file.endswith(".pdf"):
        path = os.path.join(DATASET_PATH, file)
        print("📄 Extract:", file)

        try:
            with pdfplumber.open(path) as pdf:
                full_text = ""
                for page in pdf.pages[:2]:  # ambil 2 halaman
                    full_text += page.extract_text() or ""

                text = clean_text(full_text) or ""
                text = clean_text(text)

                data[file] = {
                    "title": extract_title(text),
                    "authors": extract_authors(text),
                    "year": extract_year(text),
                    "abstract": extract_abstract(text)
                }

        except Exception as e:
            print("❌ Error:", e)


with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4)

print("✅ Extraction selesai → data.json")