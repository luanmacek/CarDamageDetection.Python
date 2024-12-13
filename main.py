import argparse
import os
import time
import datetime
import tensorflow as tf
import keras_tuner as kt
import pyodbc
import shutil
from sklearn.model_selection import train_test_split
from utils import (
    load_and_preprocess_images,
    preprocess_labels,
    save_training_model,
    save_evaluation_results,
    get_performance_info,
    initialize_database,
    get_model_id_by_path,
)
from model import build_model, evaluate_model, classes

def set_device(device):
    if device == "gpu":
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_memory_growth(gpus[0], True)
                tf.config.set_visible_devices(gpus[0], 'GPU')
                print("GPU je k dispozici a bude použito.")
            except RuntimeError as e:
                print(f"Chyba při konfiguraci GPU: {e}")
                print("Bude použito CPU.")
                tf.config.set_visible_devices([], 'GPU')  # Skrytí GPU
        else:
            print("GPU není dostupné. Bude použito CPU.")
    elif device == "cpu":
        print("Používá se CPU.")
        tf.config.set_visible_devices([], 'GPU')  # Skrytí GPU
        tf.config.set_visible_devices([tf.config.experimental.list_physical_devices('CPU')[0]], 'CPU')
    else:
        raise ValueError(f"Neplatné zařízení: {device}. Použijte 'gpu' nebo 'cpu'.")

def main():
    parser = argparse.ArgumentParser(description="Train or test a deep learning model.")
    parser.add_argument('--mode', choices=['train', 'test'], required=True, help="Mode to run the script in: 'train' or 'test'.")
    parser.add_argument('--device', choices=['gpu', 'cpu'], default='gpu', help="Device to use for training or testing: 'gpu' or 'cpu'. Default is 'gpu'.")
    parser.add_argument('--date', help="Date to specify which model to load in 'test' mode, format: YYYY-MM-DD HH-MM-SS")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for testing, default is 64.")
    args = parser.parse_args()

    set_device(args.device)
    
    conn_str = 'DRIVER={ODBC Driver 17 for SQL Server};SERVER=172.16.4.222,1433;DATABASE=CarStudent;UID=luan.macek;PWD=password;'
    conn = pyodbc.connect(conn_str)
    initialize_database(conn)

    DataDir = "images"
    train_dir = os.path.join(DataDir, 'training/')
    val_dir = os.path.join(DataDir, 'validation/')
    test_dir = os.path.join(DataDir, 'test/')
    Batchsize = 16
    Epochs = 50
    Maxepochs = 20
    
    today_date = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    model_start_date = datetime.datetime.now()

    if args.mode == 'train':
        train_data, train_labels, _ = load_and_preprocess_images(train_dir, classes)
        val_data, val_labels, _ = load_and_preprocess_images(val_dir, classes)
        train_labels, lb = preprocess_labels(train_labels)
        val_labels, _ = preprocess_labels(val_labels)

        trainX, testX, trainY, testY = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)
    
        total_start_time = time.time()
        tuner = kt.Hyperband(
            build_model,
            objective='val_accuracy',
            max_epochs=Maxepochs,
            hyperband_iterations=2,
            directory='my_dir',
            project_name='intro_to_kt'
        )

        tuner.search(trainX, trainY, epochs=Epochs, validation_data=(testX, testY))

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print(f"Best learning rate: {best_hps.get('learning_rate')}")
        print(f"Best number of units: {best_hps.get('units')}")

        model_start_time = time.time()
        model = build_model(best_hps)
        model_building_time = time.time() - model_start_time

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5, 
            restore_best_weights=True
        )

        history = model.fit(
            trainX, trainY, 
            epochs=Epochs, 
            batch_size=Batchsize, 
            validation_data=(testX, testY),
            callbacks=[early_stopping]
        )
        final_accuracy = history.history['accuracy'][-1]
        final_val_accuracy = history.history['val_accuracy'][-1]

        print(f"Final training accuracy: {final_accuracy:.4f}")
        print(f"Final validation accuracy: {final_val_accuracy:.4f}")

        total_time = time.time() - total_start_time

        model_dateId = today_date
        model_dir = f"models/{model_dateId}"

        os.makedirs(model_dir, exist_ok=True)
        shutil.copytree(train_dir, os.path.join(model_dir, 'training'))
        shutil.copytree(val_dir, os.path.join(model_dir, 'validation'))
        shutil.copytree(test_dir, os.path.join(model_dir, 'test'))
        
        path_to_model = os.path.join(model_dir, f"model_{model_dateId}.keras")
        model.save(path_to_model)

        save_training_model(
            training_date_time = model_start_date,
            total_time=round(total_time, 1),
            batch_size=Batchsize,
            epochs=Epochs,
            train_samples=len(trainX),
            val_samples=len(val_data),
            total_samples=len(testX),
            accuracy = round(final_accuracy, 3),
            val_accuracy = round(final_val_accuracy, 3),
            path_to_model=path_to_model,
            performance_info=get_performance_info(),
            conn=conn
        )
    
    if args.mode == 'test':
        test_data, test_labels, test_file_paths = load_and_preprocess_images(test_dir, classes)
        test_labels, lb = preprocess_labels(test_labels)

        model_dirs = sorted([d for d in os.listdir('models')], reverse=True)
        if not model_dirs:
            raise ValueError("No model directories found.")

        if args.date:
            matching_dirs = [d for d in model_dirs if d.startswith(args.date)]
            if not matching_dirs:
                raise ValueError(f"No model directory found for the specified date: {args.date}")
            selected_model_dir = matching_dirs[0]
        else:
            selected_model_dir = model_dirs[0]

        path_to_model = os.path.join('models', selected_model_dir, f"model_{selected_model_dir}.keras")
        print(f"Evaluation using model: {path_to_model}")
        model = tf.keras.models.load_model(path_to_model)
        model_id = get_model_id_by_path(conn, path_to_model)
        evaluation_results = evaluate_model(model, test_data, test_labels, lb, test_file_paths, args.batch_size)
        save_evaluation_results(evaluation_results, conn, model_id)

if __name__ == "__main__":
    main()