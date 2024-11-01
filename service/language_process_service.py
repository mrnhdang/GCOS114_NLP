import os
import re
import nltk
# nltk.download('punkt_tab')
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize

stopwords_file = './file/stopwords/english_stopwords'


class LanguageProcessService:

    def extract_from_file(self, file):
        data = []
        with open(file, 'r', encoding='utf8') as f:
            for line in f:
                line = line.strip()
                line = line.lower()
                data.append(line)
        return data

    def get_directory_file(self, directory):
        list_of_file = []
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            file_path = os.path.normpath(file_path)
            print(f'File name: {file_path}')

            list_of_file.append(file_path)
        return list_of_file

    def extract_stopwords(self):
        stopword = self.extract_from_file(stopwords_file)
        return stopword

    def clean_data(self, data):
        clean_data = []
        stopwords = self.extract_stopwords()
        print(f'stopwords: {stopwords}')

        # remove XML and HTML tag from text
        for text in data:
            soup = BeautifulSoup(text, "html.parser")
            clean_text = soup.get_text()

            # check empty string
            if clean_text.strip():
                # remove Punctuation
                clean_text = re.sub('\W+', ' ', clean_text)

                tokens = word_tokenize(clean_text)

                # remove stopwords
                filtered_tokens = [token for token in tokens if token not in stopwords]
                print(f'clean_text: {filtered_tokens}')
                clean_data.append(filtered_tokens)

        return clean_data

    def process_data(self, directory):
        clean_data_arr = []

        # extract data from file
        for file in self.get_directory_file(directory):
            # extract_from_file
            data = self.extract_from_file(file)

            # clean_data
            clean_data = self.clean_data(data)

            print(f'data: {clean_data}')
            clean_data_arr.append(clean_data)

        print(f'clean_data_arr: {clean_data_arr}')
