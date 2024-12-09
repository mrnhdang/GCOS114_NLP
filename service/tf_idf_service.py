import math

import numpy as np
from scipy.spatial import distance
from nltk.tokenize import word_tokenize


class TfIdfService:
    def gen_inverted_index(self, clean_data_arr, stopwords):
        inverted_index = {}
        doc_idx_token_token_freq = {}
        for data_index, documents in enumerate(clean_data_arr):
            inverted_index.update({data_index: {}})
            doc_idx_token_token_freq.update({data_index: {}})
            for doc_idx, doc in enumerate(documents):
                tokens = word_tokenize(doc.lower())
                # Tiến hành thay thế các khoảng trắng ' ' trong các từ ghép thành '_'
                tokens = [token.replace(' ', '_') for token in tokens]
                for token in tokens:
                    if token not in stopwords:
                        if token not in inverted_index[data_index].keys():
                            inverted_index[data_index][token] = [(doc_idx, 1)]
                        else:
                            is_existed = False
                            for inverted_data_idx, (target_doc_idx, target_tf) in enumerate(
                                    inverted_index[data_index][token]):
                                if target_doc_idx == doc_idx:
                                    # Tăng tần số xuất hiện của token trong tài liệu (target_doc_idx) lên 1
                                    target_tf += 1
                                    # Cập nhật lại dữ liệu
                                    inverted_index[data_index][token][inverted_data_idx] = (target_doc_idx, target_tf)
                                    is_existed = True
                                    break
                            # Trường hợp chưa tồn tại
                            if not is_existed:
                                inverted_index[data_index][token].append((doc_idx, 1))
                            if doc_idx not in doc_idx_token_token_freq[data_index].keys():
                                doc_idx_token_token_freq[data_index][doc_idx] = {}
                                doc_idx_token_token_freq[data_index][doc_idx][token] = 1
                            else:
                                if token not in doc_idx_token_token_freq[data_index][doc_idx].keys():
                                    doc_idx_token_token_freq[data_index][doc_idx][token] = 1
                                else:
                                    doc_idx_token_token_freq[data_index][doc_idx][token] += 1
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

    def calculate_tf_idf(self, inverted_index, documents):
        # Initialize TF-IDF vector as a list of dictionaries
        tfidf_vector = {}
        documents_length = len(documents)

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
        return tfidf_vector

    def calculate_cosine_similarity(self, tfidf_vector):
        # Get the number of documents
        num_documents = len(tfidf_vector)

        # Extract all unique tokens and create an index mapping
        all_tokens = set(token for doc in tfidf_vector for token in doc)
        token_to_index = {token: idx for idx, token in enumerate(all_tokens)}

        # Convert TF-IDF dictionaries into dense vectors using a matrix for efficiency
        dense_vectors = np.zeros((num_documents, len(all_tokens)))
        for doc_idx, doc in enumerate(tfidf_vector):
            for token, value in doc.items():
                dense_vectors[doc_idx][token_to_index[token]] = value

        # Precompute norms to avoid redundant calculations
        norms = np.linalg.norm(dense_vectors, axis=1)
        norms[norms == 0] = 1  # Avoid division by zero for zero vectors

        # Initialize similarity matrix
        similarity_matrix = np.eye(num_documents)  # Self-similarity is 1 (diagonal)

        # Calculate cosine similarity using vectorized operations
        for i in range(num_documents):
            for j in range(i + 1, num_documents):  # Only compute upper triangular
                similarity = 1 - distance.cosine(dense_vectors[i], dense_vectors[j])
                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity  # Mirror the value

        return similarity_matrix