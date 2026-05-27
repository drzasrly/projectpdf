import os
import pdfplumber
import re

class DocumentLoader:
    def __init__(self, folder_path, max_files=5):
        self.folder_path = folder_path
        self.max_files = max_files  

    # CLEAN TEXT
    def clean_text(self, text):
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  
        return text.strip()

    # EXTRACT TITLE
    def extract_title(self, text):
        lines = text.split("\n")

        for line in lines[:10]:
            line = line.strip()

            if len(line) < 10:
                continue

            # skip yang bukan judul
            if "journal" in line.lower():
                continue
            if "volume" in line.lower():
                continue
            if re.search(r'\d{4}', line):
                continue

            return line

        return "No Title"

    # EXTRACT ABSTRACT
    def extract_abstract(self, text):
        text_lower = text.lower()

        # regex fleksibel
        pattern = r'abstract\s*(.*?)(introduction|background|keywords)'
        match = re.search(pattern, text_lower, re.DOTALL)

        if match:
            return match.group(1).strip()

        # fallback kalau tidak ketemu
        return text[:1000]

    # LOAD FILES
    def get_as_dict(self):
        documents = {}

        if not os.path.exists(self.folder_path):
            print("Folder tidak ditemukan!")
            return documents

        files = [f for f in os.listdir(self.folder_path) if f.endswith(".pdf")]

        files = files[:self.max_files]

        for filename in files:
            filepath = os.path.join(self.folder_path, filename)

            print(f"📄 Loading: {filename}")  

            try:
                with pdfplumber.open(filepath) as pdf:
                    text = pdf.pages[0].extract_text() or ""

                text = self.clean_text(text)

                title = self.extract_title(text)
                abstract = self.extract_abstract(text)

                documents[filename] = {
                    "title": title,
                    "abstract": abstract,
                    "content": title + " " + abstract
                }

            except Exception as e:
                print(f"❌ Error di {filename}: {e}")

        print(f"✅ Total loaded: {len(documents)} dokumen")
        return documents
