# import library
from service.language_process_service import LanguageProcessService

# service injection
language_process_service = LanguageProcessService()

# static variable
stopword_file_path = './file/vietnamese-stopwords.txt'


def print_hi(name):
    str = language_process_service.extract_from_file(stopword_file_path)
    print(f'Hi, {name}')
    print(f'Hi, {str}')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
