import math

class FeatureExtraction:
    def __init__(self, documents):
        self.documents = documents
        self.vocab = sorted(set(word for doc in documents for word in doc))

    def compute_bow(self):
        bow = []
        for doc in self.documents:
            row = dict.fromkeys(self.vocab, 0)
            for word in doc:
                row[word] += 1
            bow.append(row)
        return bow

    def compute_idf(self):
        N = len(self.documents)
        idf = {}
        for word in self.vocab:
            df = sum(1 for doc in self.documents if word in doc)
            idf[word] = math.log((N / (df + 1))) + 1
        return idf

    def compute_tfidf(self):
        bow = self.compute_bow()
        idf = self.compute_idf()

        tfidf = []
        for doc in bow:
            row = {}
            for word in self.vocab:
                row[word] = doc[word] * idf[word]
            tfidf.append(row)
        return tfidf