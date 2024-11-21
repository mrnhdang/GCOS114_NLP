import math

import numpy as np
from scipy.spatial import distance
from nltk.tokenize import word_tokenize


class TfIdfService:
    def gen_inverted_index(self, documents):
        inverted_index = {}
        doc_idx_token_token_freq = {}
        for doc_idx, doc in enumerate(documents):
            tokens = word_tokenize(doc.lower())
            # Tiến hành thay thế các khoảng trắng ' ' trong các từ ghép thành '_'
            tokens = [token.replace(' ', '_') for token in tokens]
            for token in tokens:
                if token not in inverted_index.keys():
                    inverted_index[token] = [(doc_idx, 1)]
                else:
                    is_existed = False
                    for inverted_data_idx, (target_doc_idx, target_tf) in enumerate(inverted_index[token]):
                        if target_doc_idx == doc_idx:
                            # Tăng tần số xuất hiện của token trong tài liệu (target_doc_idx) lên 1
                            target_tf += 1
                            # Cập nhật lại dữ liệu
                            inverted_index[token][inverted_data_idx] = (target_doc_idx, target_tf)
                            is_existed = True
                            break
                    # Trường hợp chưa tồn tại
                    if not is_existed:
                        inverted_index[token].append((doc_idx, 1))
                    if doc_idx not in doc_idx_token_token_freq.keys():
                        doc_idx_token_token_freq[doc_idx] = {}
                        doc_idx_token_token_freq[doc_idx][token] = 1
                    else:
                        if token not in doc_idx_token_token_freq[doc_idx].keys():
                            doc_idx_token_token_freq[doc_idx][token] = 1
                        else:
                            doc_idx_token_token_freq[doc_idx][token] += 1
        return inverted_index, doc_idx_token_token_freq

    def find_max_freq_token(self, doc_idx, doc_idx_token_token_freq):
        max_freq_token = ''
        max_freq = 0
        if doc_idx in doc_idx_token_token_freq.keys():
            for token in doc_idx_token_token_freq[doc_idx].keys():
                if doc_idx_token_token_freq[doc_idx][token] > max_freq:
                    max_freq_token = token
                    max_freq = doc_idx_token_token_freq[doc_idx][token]
        return max_freq_token, max_freq

    def gen_doc_idx_max_freq_token(self, documents, doc_idx_token_token_freq):
        doc_idx_max_freq_token = {}
        for doc_idx, doc in enumerate(documents):
            doc_idx_max_freq_token[doc_idx] = self.find_max_freq_token(doc_idx, doc_idx_token_token_freq)
        return doc_idx_max_freq_token

    def calculate_tf_idf(self, inverted_index, documents_length):
        # Initialize TF-IDF vector as a list of dictionaries
        tfidf_vector = [{} for _ in range(documents_length)]

        # Calculate TF-IDF for each token
        for token, D_t in inverted_index.items():
            doc_freq = len(D_t)
            idf = math.log((documents_length / doc_freq), 2)

            for doc_idx, tf in D_t:
                tf_idf = tf * idf
                if token in tfidf_vector[doc_idx]:
                    tfidf_vector[doc_idx][token] += tf_idf
                else:
                    tfidf_vector[doc_idx][token] = tf_idf
        # print(f'tfidf_vector: {tfidf_vector}')
        return tfidf_vector

    def calculate_cosine_similarity(self, tfidf_vector):
        # Get the number of documents
        num_documents = len(tfidf_vector)

        # Initialize a similarity matrix with zeros
        similarity_matrix = np.zeros((num_documents, num_documents))

        # Convert TF-IDF dictionaries into dense vectors
        all_tokens = set(token for doc in tfidf_vector for token in doc.keys())
        token_to_index = {token: idx for idx, token in enumerate(all_tokens)}

        dense_vectors = []
        for doc in tfidf_vector:
            vector = [doc.get(token, 0) for token in token_to_index]
            dense_vectors.append(vector)

        # Calculate cosine similarity
        for i in range(num_documents):
            for j in range(num_documents):
                if i == j:
                    similarity_matrix[i][j] = 1.0  # Self-similarity is always 1
                else:
                    similarity_matrix[i][j] = 1 - distance.cosine(dense_vectors[i], dense_vectors[j])

        return similarity_matrix
