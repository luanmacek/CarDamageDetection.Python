import os
import time
import numpy as np
import tensorflow as tf
import pyodbc
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
import json
import re
from datetime import datetime

def load_and_preprocess_images(data_dir, classes, img_size=(224, 224)):
    data = []
    labels = []
    file_paths = []
    for class_ in classes:
        path = os.path.join(data_dir, class_)
        if not os.path.exists(path):
            continue
        for image in os.listdir(path):
            image_path = os.path.join(path, image)
            try:
                image_ = load_img(image_path, target_size=img_size)
                image_ = img_to_array(image_)
                image_ = tf.keras.applications.mobilenet_v2.preprocess_input(image_)
                data.append(image_)
                labels.append(class_)
                file_paths.append(image_path)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
    return np.array(data, dtype="float32"), labels, file_paths

def preprocess_labels(labels):
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = to_categorical(labels)  
    return labels, lb

def save_model_and_get_id(model, conn):
    cursor = conn.cursor()
    cursor.execute("INSERT INTO TrainingModels (TrainingDateTime) VALUES (?)", (time.strftime('%Y-%m-%d %H:%M:%S'),))
    model_id = cursor.execute("SELECT SCOPE_IDENTITY()").fetchone()[0]
    conn.commit()
    cursor.close()

    path_to_model = f"models/{model_id}/{model_id}.keras"
    os.makedirs(os.path.dirname(path_to_model), exist_ok=True)
    model.save(path_to_model)
    
    return model_id

def save_training_model(
    training_date_time, 
    total_time,                
    batch_size, 
    epochs, 
    train_samples, 
    val_samples, 
    total_samples,
    accuracy,
    val_accuracy, 
    path_to_model, 
    performance_info, 
    conn
):
    cursor = conn.cursor()   
    performance_info_json = json.dumps(performance_info)
    
    query = """
    INSERT INTO TrainingModels 
    ( TrainingDateTime, TotalTIme, BatchSize, Epochs, TrainSamples, ValidationSamples, TotalSamples,TrainingAccuracy, ValidationAccuracy, ModelPath,PerformanceInfo)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    values = (
        training_date_time, 
        total_time,  
        batch_size, 
        epochs, 
        train_samples, 
        val_samples, 
        total_samples, 
        accuracy,
        val_accuracy,
        path_to_model, 
        performance_info_json
    )
    
    cursor.execute(query, values)
    conn.commit()
    cursor.close()

def save_evaluation_results(evaluation_results, conn, model_id):
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM TrainingModels WHERE ModelID = ?", (model_id,))
    if not cursor.fetchone():
        print(f"ModelID {model_id} does not exist in TrainingModels.")
        return

    insert_query = """ INSERT INTO TestModels (
            ModelID, 
            CategoryCount, 
            BatchSize,
            Accuracy, 
            Precision, 
            OverallAccuracy, 
            EvaluationDuration
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """

    print(f"Saving evaluation results for model_id: {model_id}")

    category_count = len(set(evaluation_results['correct_categories'])) 
    batch_size = evaluation_results['batch_size']
    overall_accuracy = evaluation_results['overall_accuracy']
    avg_accuracy = round((evaluation_results['accuracy_damaged'] + evaluation_results['accuracy_ok']) / 2, 2)
    avg_precision = round((evaluation_results['precision_damaged'] + evaluation_results['precision_ok']) / 2, 2)
    evaluation_duration = round(sum(evaluation_results['execution_time']) / 1000, 2) 

    try:
        cursor.execute(insert_query, (
            model_id, 
            category_count,
            batch_size,
            avg_accuracy, 
            avg_precision, 
            overall_accuracy, 
            evaluation_duration,
        ))
        conn.commit()
        print("SUCCESS")
    except pyodbc.Error as e:
        print(f"Error inserting evaluation results: {e}")
        conn.rollback()



def initialize_database(conn):
    cursor = conn.cursor()

    try:
        cursor.execute("""
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='TrainingModels' AND xtype='U')
        CREATE TABLE dbo.TrainingModels (
            ModelID INT IDENTITY(1,1) PRIMARY KEY,
            TrainingDateTime DATETIME,   
            TotalTile FLOAT,              
            BatchSize INT,
            Epochs INT,
            TrainSamples INT,
            ValSamples INT,
            TotalSamples INT,
            TrainingAccuracy FLOAT, 
            ValidationAccuracy FLOAT,
            ModelPath VARCHAR(255),
            PerformanceInfo TEXT
        )
        """)
        conn.commit()
    except pyodbc.Error as e:
        print(f"Error creating TrainingModels table: {e}")

    try:
        cursor.execute("""
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='TestImages' AND xtype='U')
        CREATE TABLE dbo.TestImages (
            EvaluationID INT IDENTITY(1,1) PRIMARY KEY,
            ImagePath VARCHAR(255),
            ModelID INT,
            CorrectCategory VARCHAR(255),
            PredictedCategory VARCHAR(255),
            IsCorrect BIT,
            Prediction FLOAT,
            Score FLOAT,
            ExecutionTime FLOAT,  
            FOREIGN KEY (ModelID) REFERENCES TrainingModels(ModelID)
        )
        """)
        conn.commit()
    except pyodbc.Error as e:
        print(f"Error creating TestImages table: {e}")

    try:
        cursor.execute("""
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='TestModels' AND xtype='U')
        CREATE TABLE dbo.TestModels (
            EvaluationID INT IDENTITY(1,1) PRIMARY KEY,
            ModelID INT,
            CategoryCount VARCHAR(255),
            Accuracy VARCHAR(255),
            Precision VARCHAR(255),
            OverallAccuracy FLOAT,
            EvaluationDuration FLOAT,  
            FOREIGN KEY (ModelID) REFERENCES TrainingModels(ModelID)
        )
        """)
        conn.commit()
    except pyodbc.Error as e:
        print(f"Error creating TestModels table: {e}")

    cursor.close()

def get_performance_info():
    if tf.config.list_physical_devices('GPU'):
        device_type = 'GPU'
        device_name = tf.config.list_physical_devices('GPU')[0].name
        try:
            memory_info = tf.config.experimental.get_memory_info('GPU:0')
            performance_info = {
                'device_type': device_type,
                'device_name': device_name,
                'memory_limit': memory_info.get('total', None),
                'memory_usage': memory_info.get('current', None),
            }
        except Exception as e:
            print(f"Error getting GPU memory info: {e}")
            performance_info = {
                'device_type': device_type,
                'device_name': device_name,
                'memory_limit': None,
                'memory_usage': None,
            }
    else:
        device_type = 'CPU'
        device_name = 'CPU'
        performance_info = {
            'device_type': device_type,
            'device_name': device_name,
            'memory_limit': None,
            'memory_usage': None,
        }

    return performance_info

def filename_to_datetime(filename):
    date_str = filename.split('_')[1] 
    time_str = filename.split('_')[2].replace('.keras', '')  
    datetime_str = f"{date_str} {time_str.replace('-', ':')}"
    
    try:
        dt = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
    except ValueError as e:
        print(f"Error parsing datetime: {e}")
        return None
    
    return dt

def get_model_id_by_path(conn, path_to_model):
    cursor = conn.cursor()
    query = """
    SELECT ModelID FROM TrainingModels
    WHERE ModelPath = ?
    """
    try:
        cursor.execute(query, (path_to_model,))
        result = cursor.fetchone()
        if result:
            return result[0]
        else:
            print(f"No model found for path: {path_to_model}")
            return None
    except pyodbc.Error as e:
        print(f"Error fetching model ID: {e}")
        return None
    finally:
        cursor.close()