import datetime
import os
import cv2
import random
import logging
import pandas as pd
import tensorflow as tf
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
TRAIN_COUNT = 200000
VALIDATION_COUNT = 10000
CONTROL_COUNT = 19000
BATCH_SIZE = 2048
EPOCHS = 30
EPOCHS_RANGE = range(EPOCHS)
IMAGE_LENGTH = 28
IMAGE_HEIGHT = 28
ACTIVATION = 'relu'  # 'sigmoid'
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
    #logging.info('Данные отображены')


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
    #show_histogram(classes_images_counts, 'Гистограмма количества изображений классов')
    #logging.info('Гистограмма количества изображений классов отображена')
    #show_histogram(classes_images_percents, 'Гистограмма частоты классов')
    #logging.info('Гистограмма частоты классов отображена')


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
    data = list(data_frame[DATA_COLUMN_NAME].values)
    labels = list(data_frame[LABELS_COLUMN_NAME].values)

    data_dataset = tf.data.Dataset.from_tensor_slices(data)
    labels_dataset = tf.data.Dataset.from_tensor_slices(labels)
    dataset = tf.data.Dataset.zip((data_dataset, labels_dataset))

    train_dataset = dataset.take(TRAIN_COUNT).batch(BATCH_SIZE)
    validation_dataset = dataset.skip(TRAIN_COUNT).take(VALIDATION_COUNT).batch(BATCH_SIZE)
    control_dataset = dataset.skip(TRAIN_COUNT + VALIDATION_COUNT).take(CONTROL_COUNT).batch(BATCH_SIZE)

    logging.info('Данные разделены на три выборки: обучающую, валидационную и контрольную (тестовую)')

    return train_dataset, validation_dataset, control_dataset


def get_statistics(model, train_dataset, validation_dataset, control_dataset, optimizer, log):
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    model_history = model.fit(
        x=train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        verbose=1
    )

    loss, accuracy = model.evaluate(control_dataset)
    logging.info(f'{log}, accuracy = {accuracy}, loss = {loss}')

    accuracy = model_history.history['accuracy']
    validation_accuracy = model_history.history['val_accuracy']
    loss = model_history.history['loss']
    validation_loss = model_history.history['val_loss']

    return loss, accuracy, validation_loss, validation_accuracy


def get_neural_network_statistics(train_dataset, validation_dataset, control_dataset):
    losses = list()
    accuracies = list()
    validation_losses = list()
    validation_accuracies = list()

    train_dataset = train_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    control_dataset = control_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    simple_model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1. / 255),
        tf.keras.layers.Flatten(input_shape=(IMAGE_LENGTH, IMAGE_HEIGHT)),
        tf.keras.layers.Dense(512, activation=ACTIVATION),
        tf.keras.layers.Dense(512, activation=ACTIVATION),
        tf.keras.layers.Dense(256, activation=ACTIVATION),
        tf.keras.layers.Dense(256, activation=ACTIVATION),
        tf.keras.layers.Dense(128, activation=ACTIVATION),
        tf.keras.layers.Dense(CLASSES_COUNT)
    ])

    regularized_model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1. / 255),
        tf.keras.layers.Flatten(input_shape=(IMAGE_LENGTH, IMAGE_HEIGHT)),
        tf.keras.layers.Dense(512, activation=ACTIVATION, kernel_regularizer=tf.keras.regularizers.L2(0.001)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation=ACTIVATION, kernel_regularizer=tf.keras.regularizers.L2(0.001)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation=ACTIVATION, kernel_regularizer=tf.keras.regularizers.L2(0.001)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(256, activation=ACTIVATION, kernel_regularizer=tf.keras.regularizers.L2(0.001)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(128, activation=ACTIVATION, kernel_regularizer=tf.keras.regularizers.L2(0.001)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(CLASSES_COUNT)
    ])

    dynamic_rate_model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1. / 255),
        tf.keras.layers.Flatten(input_shape=(IMAGE_LENGTH, IMAGE_HEIGHT)),
        tf.keras.layers.Dense(512, activation=ACTIVATION, kernel_regularizer=tf.keras.regularizers.L2(0.001)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation=ACTIVATION, kernel_regularizer=tf.keras.regularizers.L2(0.001)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation=ACTIVATION, kernel_regularizer=tf.keras.regularizers.L2(0.001)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(256, activation=ACTIVATION, kernel_regularizer=tf.keras.regularizers.L2(0.001)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(128, activation=ACTIVATION, kernel_regularizer=tf.keras.regularizers.L2(0.001)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(CLASSES_COUNT)
    ])

    simple_model_statistics = get_statistics(
        simple_model, train_dataset, validation_dataset, control_dataset, tf.keras.optimizers.experimental.SGD(), 'Простая модель'
    )
    regularized_model_statistics = get_statistics(
        regularized_model, train_dataset, validation_dataset, control_dataset, tf.keras.optimizers.experimental.SGD(), 'Модель с регуляризацией и сбросом нейронов'
    )
    dynamic_rate_model_statistics = get_statistics(
        dynamic_rate_model, train_dataset, validation_dataset, control_dataset, tf.keras.optimizers.Adam(learning_rate=0.001), 'Модель с динамической скоростью обучения'
    )

    losses.extend((simple_model_statistics[0], regularized_model_statistics[0], dynamic_rate_model_statistics[0]))
    accuracies.extend((simple_model_statistics[1], regularized_model_statistics[1], dynamic_rate_model_statistics[1]))
    validation_losses.extend((simple_model_statistics[2], regularized_model_statistics[2], dynamic_rate_model_statistics[2]))
    validation_accuracies.extend((simple_model_statistics[3], regularized_model_statistics[3], dynamic_rate_model_statistics[3]))

    return losses, accuracies, validation_losses, validation_accuracies


def show_result_plot(losses, accuracies, validation_losses, validation_accuracies):
    plt.figure(figsize=(15, 8))

    plt.subplot(1, 2, 1)
    plt.title('Точность')
    plt.plot(EPOCHS_RANGE, accuracies[0], label='Train Simple')
    plt.plot(EPOCHS_RANGE, validation_accuracies[0], label='Valid Simple', linestyle='dotted')
    plt.plot(EPOCHS_RANGE, accuracies[1], label='Train Regular')
    plt.plot(EPOCHS_RANGE, validation_accuracies[1], label='Valid Regular', linestyle='dotted')
    plt.plot(EPOCHS_RANGE, accuracies[2], label='Train Dynamic')
    plt.plot(EPOCHS_RANGE, validation_accuracies[2], label='Valid Dynamic', linestyle='dotted')
    plt.legend(loc='lower right')

    plt.subplot(1, 2, 2)
    plt.title('Потери')
    plt.plot(EPOCHS_RANGE, losses[0], label='Train Simple')
    plt.plot(EPOCHS_RANGE, validation_losses[0], label='Valid Simple', linestyle='dotted')
    plt.plot(EPOCHS_RANGE, losses[1], label='Train Regular')
    plt.plot(EPOCHS_RANGE, validation_losses[1], label='Valid Regular', linestyle='dotted')
    plt.plot(EPOCHS_RANGE, losses[2], label='Train Dynamic')
    plt.plot(EPOCHS_RANGE, validation_losses[2], label='Valid Dynamic', linestyle='dotted')
    plt.legend(loc='upper right')

    plt.show()
    logging.info('Графики потерь и точности показаны')


def main():
    start_time = datetime.datetime.now()

    # Задание 1
    data_frame = create_data_frame()
    #show_images()
    # Задание 2
    check_classes_balance(data_frame)
    # Задание 4
    data_frame = remove_duplicates(data_frame)
    # Задание 2
    check_classes_balance(data_frame)
    # Задание 3
    data_frame = mix_dataset(data_frame)
    train_dataset, validation_dataset, test_dataset = split_dataset(data_frame)
    # ЛР 2
    losses, accuracies, validation_losses, validation_accuracies = get_neural_network_statistics(
        train_dataset, validation_dataset, test_dataset
    )
    show_result_plot(losses, accuracies, validation_losses, validation_accuracies)

    end_time = datetime.datetime.now()
    logging.info(f'Время работы программы: {end_time - start_time}')


main()
