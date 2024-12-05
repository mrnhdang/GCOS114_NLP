import os
import re
import nltk
import numpy as np
# nltk.download('punkt_tab')
from bs4 import BeautifulSoup
# from nltk.metrics.aline import similarity_matrix
from nltk.tokenize import word_tokenize

from service.tf_idf_service import TfIdfService
from service.page_rank_service import PageRankService
from service.rough_service import RoughService

stopwords_file = './file/stopwords/english_stopwords'
tf_idf_service = TfIdfService()
page_rank_service = PageRankService()
rough_service = RoughService()


class LanguageProcessService:

    def extract_from_file(self, file):
        data = []
        with open(file, 'r', encoding='utf8') as f:
            for line in f:
                line = line.strip()
                data.append(line)
        return data

    def get_directory_file(self, directory):
        list_of_file = []
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            file_path = os.path.normpath(file_path)
            list_of_file.append(file_path)
        return list_of_file

    def extract_stopwords(self):
        stopword = self.extract_from_file(stopwords_file)
        return stopword

    def clean_data(self, data):
        clean_data = []
        clean_text_arr = []

        # remove XML and HTML tag from text
        for text in data:
            soup = BeautifulSoup(text, "html.parser")
            clean_text = soup.get_text()

            # check empty string
            if clean_text.strip():
                clean_text_arr.append(clean_text)

                # remove Punctuation
                clean_text = re.sub('\W+', ' ', clean_text)
                clean_text = clean_text.lower()
                clean_data.append(clean_text)
        return clean_data, clean_text_arr

    def process_data(self, directory):
        clean_data_arr = []
        clean_text_arr = []
        data = []

        # extract data from file
        for file in self.get_directory_file(directory):
            # extract_from_file
            temp = self.extract_from_file(file)
            data.append(temp)

        # clean_data
        for text in data:
            clean_data, clean_text = self.clean_data(text)
            clean_data_arr.append(clean_data)
            clean_text_arr.append(clean_text)

        stopwords = self.extract_stopwords()

        inverted_index, doc_idx_token_token_freq = tf_idf_service.gen_inverted_index(clean_data_arr, stopwords)

        doc_idx_max_freq_token = tf_idf_service.gen_doc_idx_max_freq_token(clean_data_arr,
                                                                           doc_idx_token_token_freq)
        for data_index, documents in enumerate(clean_data_arr):
            for token in inverted_index[data_index].keys():
                D_t = inverted_index[data_index][token]
                for inverted_data_idx, (doc_idx, tf) in enumerate(D_t):
                    # Cập nhật lại trọng số tf của token đang xét
                    (max_freq_token, max_freq) = doc_idx_max_freq_token[data_index][doc_idx]
                    if max_freq > 0:
                        update_tf = tf / max_freq
                        # Cập nhật lại dữ liệu
                        inverted_index[data_index][token][inverted_data_idx] = (doc_idx, update_tf)

        tfidf_vector = tf_idf_service.calculate_tf_idf(inverted_index, clean_data_arr)

        similarity_matrix = tf_idf_service.calculate_cosine_similarity(tfidf_vector, clean_data_arr)

        pagerank_scores = page_rank_service.cal_page_rank_score(similarity_matrix, clean_data_arr)

        # print(clean_data_arr)
        summary_arr = []
        for data_index, documents in enumerate(clean_data_arr):
            length = len(clean_text_arr[data_index])
            k = int(length * 10 / 100)
            ranked_sentences = sorted(
                ((score, idx) for idx, score in enumerate(pagerank_scores[data_index])),
                reverse=True
            )
            summary = ''
            for _, idx in ranked_sentences[:k]:
                summary = summary + ' ' + clean_text_arr[data_index][idx]
            summary_arr.append(summary)
            # print(f'\nSummary:\n{summary}')
        return summary_arr
