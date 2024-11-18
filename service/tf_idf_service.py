import math

import numpy as np
from nltk.metrics.aline import similarity_matrix
from scipy.spatial import distance


class TfIdfService:
    def gen_inverted_index(self, documents):
        inverted_index = {}
        doc_idx_token_token_freq = {}
        for doc_idx, doc in enumerate(documents):
            for token in doc:
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
                    if is_existed == False:
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
        return (max_freq_token, max_freq)

    def gen_doc_idx_max_freq_token(self, documents, doc_idx_token_token_freq):
        doc_idx_max_freq_token = {}
        for doc_idx, doc in enumerate(documents):
            doc_idx_max_freq_token[doc_idx] = self.find_max_freq_token(doc_idx, doc_idx_token_token_freq)
        return doc_idx_max_freq_token

    def calculate_tf_idf(self, inverted_index, documents_length):
        # print(f'inverted_index: {inverted_index}')
        doc_id_tfidf_encoded_vector_dict = {}
        for token in inverted_index.keys():
            D_t = inverted_index[token]
            doc_freq = len(D_t)
            tfidf_vector = np.zeros(documents_length)
            for index, (doc_idx, tf) in enumerate(D_t):
                idf = math.log((documents_length / doc_freq), 2)
                tfidf = tf * idf
                # print('Từ: [', token, ']-> tài liệu số: [', doc_idx, '], TF-IDF = [', tfidf, ']')

                tfidf_vector[index] = tfidf
                doc_id_tfidf_encoded_vector_dict[doc_idx] = tfidf_vector
        # print(f'doc_id_tfidf_encoded_vector_dict: {doc_id_tfidf_encoded_vector_dict}/n')

        # Duyệt qua danh sách các tài liệu/văn bản (đã mã hóa ở dạng  vectors)
        similarity_matrix = np.zeros((documents_length, documents_length))

        for i in range(documents_length):
            for j in range(documents_length):
                if i != j:
                    similarity_matrix[i][j] = 1 - distance.cosine(doc_id_tfidf_encoded_vector_dict[i],
                                                                  doc_id_tfidf_encoded_vector_dict[j])

        # print(f'similarity_matrix:\n {similarity_matrix[0]}')
        return similarity_matrix
