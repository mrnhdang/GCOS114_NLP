# import library
from service.language_process_service import LanguageProcessService, rough_service
from service.rough_service import RoughService

# service injection
language_process_service = LanguageProcessService()
rough_service = RoughService()

# static variable
stopwords_dir = './file/stopwords'
test_dir = './file/test'
train_dir = './file/train'


def main_function():
    summary = language_process_service.process_data(test_dir)

    # calculate score between output data and provided result
    sum_data = language_process_service.extract_from_file("./file/sum/d061j")
    _, sum_text_clean = language_process_service.clean_data(sum_data)
    sum_data_text = ' '.join(text for text in sum_text_clean)

    print(f'SUM DATA TEXT: \n {sum_data_text}')
    scores = rough_service.calculate_rough(sum_data_text, summary)
    print(f'\n')
    print(f'scores:\n {scores}')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main_function()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
