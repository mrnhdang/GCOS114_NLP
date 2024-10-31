import os

stopwords_dir = './file/stopwords'

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
            print(f'File name: {file_path}')

            list_of_file.append(file_path)
        return list_of_file

    def extract_stopwords(self):
        data = []
        dir1 = self.get_directory_file(stopwords_dir)
        for file in dir1:
            stopword = self.extract_from_file(file)
            print(f'stopword: {stopword}')

            data.append(stopword)
        return data

    def clean_data(self, data):
        stopword = self.extract_stopwords()


    def process_data(self, directory):
        dir3 = self.get_directory_file(directory)
        for file in dir3:
            data = self.extract_from_file(file)
            print(f'data: {data}')