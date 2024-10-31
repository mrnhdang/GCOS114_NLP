# import library
from service.language_process_service import LanguageProcessService

# service injection
language_process_service = LanguageProcessService()

# static variable
stopwords_dir = './file/stopwords'
test_dir = './file/test'
train_dir = './file/train'


def main_function():
    language_process_service.process_data(test_dir)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main_function()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
