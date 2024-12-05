# import library
from service.language_process_service import LanguageProcessService, rough_service
from service.rough_service import RoughService

# service injection
language_process_service = LanguageProcessService()
rough_service = RoughService()

# static variable
stopwords_dir = './file/stopwords'
train_dir = './file/train'


def main_function():
    train_arr = language_process_service.process_data(train_dir)

    # calculate score between output data and provided result
    sum_arr = []
    for file in language_process_service.get_directory_file("./file/sum"):
        data = language_process_service.extract_from_file(file)
        _, sum_text_clean = language_process_service.clean_data(data)
        sum_data_text = ' '.join(text for text in sum_text_clean)
        sum_arr.append(sum_data_text)

    for index, train in enumerate(train_arr):
        scores = rough_service.calculate_rough(sum_arr[index], train)
        print(f'\n Summary:\n{train}')
        print(f'Score: {scores}')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main_function()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
