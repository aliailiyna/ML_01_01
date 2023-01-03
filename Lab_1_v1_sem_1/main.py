import datetime
import os
import cv2
import random
import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt

DATASET_PATH = os.path.join('D:' + os.sep, 'notMNIST_large')
CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
DATA_COLUMN_NAME = 'data'
LABELS_COLUMN_NAME = 'labels'
HASHED_DATA_COLUMN_NAME = 'hashes'
CLASSES_COUNT = len(CLASSES)
BALANCE = 1 / CLASSES_COUNT
BALANCE_BORDER_LOW = BALANCE * 0.88
BALANCE_BORDER_HIGH = BALANCE * 1.12
#MAX_ITERATIONS_COUNT = 200000
MAX_ITERATIONS_COUNT = 100
#TRAIN_SIZES = [50, 100, 1000, 50000]
#TRAIN_SIZES = [50, 100, 500, 1000]
TRAIN_SIZES = [50, 100, 500, 1000, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000]
TRAIN_COUNT = 200000
VALIDATION_COUNT = 10000
CONTROL_COUNT = 19000
#logging.basicConfig(level=logging.INFO, format='%(levelname)s %(asctime)s %(message)s')
logging.basicConfig(filename='log.txt', level=logging.INFO, format='%(levelname)s %(asctime)s %(message)s')


def get_single_class_data(folder_path):  # Задание 1
    result_data = list()
    files = os.listdir(folder_path)
    for file in files:
        image_path = os.path.join(folder_path, file)
        img = cv2.imread(image_path)
        if img is not None:
            result_data.append(img.reshape(-1))

    return result_data


def create_data_frame():  # Задание 1
    data = list()
    labels = list()
    for class_item in CLASSES:
        class_folder_path = os.path.join(DATASET_PATH, class_item)
        class_data = get_single_class_data(class_folder_path)

        data.extend(class_data)
        labels.extend([CLASSES.index(class_item) for _ in range(len(class_data))])

    data_frame = pd.DataFrame({DATA_COLUMN_NAME: data, LABELS_COLUMN_NAME: labels})
    logging.info('Данные загружены')

    return data_frame


def show_images():  # Задание 1
    pictures = list()
    for class_item in CLASSES:
        image_folder = os.path.join(DATASET_PATH, class_item)
        file = random.choice(os.listdir(image_folder))
        image_path = os.path.join(image_folder, file)
        pictures.append(image_path)
    pic_box = plt.figure(figsize=(10, 5))
    for i, picture in enumerate(pictures):
        picture = cv2.imread(picture)
        pic_box.add_subplot(2, 5, i + 1)
        plt.imshow(picture)
        plt.title(CLASSES[i])
        plt.axis('off')
    # вывод всех изображений на экран
    plt.show()
    logging.info('Данные отображены')


def get_classes_images_counts(data_frame):  # Задание 2
    classes_images_counts = list()
    for class_index in range(len(CLASSES)):
        labels = data_frame[LABELS_COLUMN_NAME]
        class_rows = data_frame[labels == class_index]
        class_count = len(class_rows)

        classes_images_counts.append(class_count)

    return classes_images_counts


def show_histogram(classes_images, title):  # Задание 2
    plt.figure()
    plt.bar(CLASSES, classes_images)
    plt.title(title)
    plt.show()


def check_classes_balance(data_frame):  # Задание 2
    classes_images_counts = get_classes_images_counts(data_frame)
    classes_images_percents = list()
    images_sum = sum(classes_images_counts)
    logging.info(f'Общее количество изображений {images_sum}')
    is_balanced = True
    for count in classes_images_counts:
        percent = count / images_sum
        classes_images_percents.append(percent)
        if not (BALANCE_BORDER_LOW < percent < BALANCE_BORDER_HIGH):
            is_balanced = False

    for class_index in range(CLASSES_COUNT):
        logging.info(f'Класс {CLASSES[class_index]}. Количество изображений: {classes_images_counts[class_index]}. Частота: {classes_images_percents[class_index]:.20f}')
    if is_balanced:
        logging.info('Классы сбалансированы')
    else:
        logging.info('Классы не сбалансированы')
    show_histogram(classes_images_counts, 'Гистограмма количества изображений классов')
    logging.info('Гистограмма количества изображений классов отображена')
    show_histogram(classes_images_percents, 'Гистограмма частоты классов')
    logging.info('Гистограмма частоты классов отображена')


def remove_duplicates(data):  # Задание 4
    data_bytes = [item.tobytes() for item in data[DATA_COLUMN_NAME]]
    data[HASHED_DATA_COLUMN_NAME] = data_bytes
    data.sort_values(HASHED_DATA_COLUMN_NAME, inplace=True)
    data.drop_duplicates(subset=HASHED_DATA_COLUMN_NAME, keep='first', inplace=True)
    data.pop(HASHED_DATA_COLUMN_NAME)
    logging.info('Дубликаты удалены')

    return data


def mix_dataset(data):  # Задание 3
    data_mixed = data.sample(frac=1, random_state=18)
    logging.info('Данные перемешаны')

    return data_mixed


def split_dataset(data_frame):  # Задание 3
    train_single_class = int(TRAIN_COUNT / CLASSES_COUNT)
    validation_single_class = int(VALIDATION_COUNT / CLASSES_COUNT)
    test_single_class = int(CONTROL_COUNT / CLASSES_COUNT)
    train = data_frame.apply(lambda x: x[:train_single_class])
    validation = data_frame.apply(lambda x: x[train_single_class : train_single_class + validation_single_class])
    control = data_frame.apply(lambda x: x[train_single_class + validation_single_class : train_single_class + validation_single_class + test_single_class])

    data_train = np.array(list(train[DATA_COLUMN_NAME].values), np.float32)
    labels_train = np.array(list(train[LABELS_COLUMN_NAME].values), np.float32)
    data_validation = np.array(list(validation[DATA_COLUMN_NAME].values), np.float32)
    labels_validation = np.array(list(validation[LABELS_COLUMN_NAME].values), np.float32)
    data_control = np.array(list(control[DATA_COLUMN_NAME].values), np.float32)
    labels_control = np.array(list(control[LABELS_COLUMN_NAME].values), np.float32)


    logging.info('Данные разделены на три выборки: обучающую, валидационную и контрольную (тестовую)')
    return data_train, labels_train, data_control, labels_control, data_validation, labels_validation


def logistic_regression(data_train, labels_train, data_control, labels_control):  # Задание 5
    accuracies = list()
    for train_size in TRAIN_SIZES:
        logistic_regression = LogisticRegression(max_iter=MAX_ITERATIONS_COUNT)
        logistic_regression.fit(data_train[:train_size], labels_train[:train_size])
        #logging.info('Классификатор с помощью линейной регрессии обучен')

        score = logistic_regression.score(data_control, labels_control)
        #logging.info('Точность рассчитана')
        accuracies.append(score)
        logging.info(f'Точность для размера обучающей выборки {train_size} - {score}')

    #for accuracy_index in range(len(accuracies)):
        #logging.info(f'Точность для размера обучающей выборки {TRAIN_SIZES[accuracy_index]} - {accuracies[accuracy_index]}')

    return accuracies


def show_accuracies_plot(accuracies):  # Задание 5
    plt.figure()
    plt.title('График зависимости точности классификатора от размера обучающей выборки')
    plt.xlabel('Размер обучающей выборки')
    plt.ylabel('Точность')
    plt.grid()

    #plt.plot(TRAIN_SIZES, accuracies, 'o-', color='g', label='Testing score')
    plt.plot(TRAIN_SIZES, accuracies)

    plt.show()
    logging.info('График зависимости точности классификатора отображен')


def main():
    start_time = datetime.datetime.now()

    # Задание 1
    data_frame = create_data_frame()
    show_images()
    # Задание 2
    check_classes_balance(data_frame)
    # Задание 4
    data_frame = remove_duplicates(data_frame)
    # Задание 2
    check_classes_balance(data_frame)
    # Задание 3
    data_frame = mix_dataset(data_frame)
    data_train, labels_train, data_control, labels_control, *_ = split_dataset(data_frame)
    # Задание 5
    accuracies = logistic_regression(data_train, labels_train, data_control, labels_control)
    show_accuracies_plot(accuracies)

    end_time = datetime.datetime.now()
    logging.info(f'Время работы программы: {end_time - start_time}')


main()
