import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras.layers import Dense, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import time

classes = ["Damaged", "Ok"]

def build_model(hp):
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_tensor=tf.keras.Input(shape=(224, 224, 3)))
    x = base_model.output
    x = MaxPooling2D(pool_size=(5, 5))(x)
    x = Flatten(name="flatten")(x)
    x = Dense(hp.Int('units', min_value=64, max_value=256, step=64), activation="relu")(x)
    x = Dropout(hp.Float('dropout', min_value=0.3, max_value=0.7, step=0.1))(x)
    x = Dense(len(classes), activation="softmax")(x)
    
    model = Model(inputs=base_model.input, outputs=x)
    
    for layer in base_model.layers:
        layer.trainable = hp.Boolean('fine_tune')
    
    model.compile(
        optimizer=Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def evaluate_model(model, test_data, test_labels, lb, file_paths, batch_size):
    start_time = time.time()
    predictions = model.predict(test_data, batch_size=32)
    prediction_times = [time.time() - start_time] * len(test_data)

    predict_index = np.argmax(predictions, axis=1)
    true_classes = np.argmax(test_labels, axis=1)

    tp_damaged = np.sum((true_classes == 0) & (predict_index == 0))
    fp_damaged = np.sum((true_classes != 0) & (predict_index == 0))
    fn_damaged = np.sum((true_classes == 0) & (predict_index != 0))

    tp_ok = np.sum((true_classes == 1) & (predict_index == 1))
    fp_ok = np.sum((true_classes != 1) & (predict_index == 1))
    fn_ok = np.sum((true_classes == 1) & (predict_index != 1))

    total_samples = len(true_classes)
    damaged_total = np.sum(true_classes == 0)
    ok_total = np.sum(true_classes == 1)

    damaged_accuracy = (tp_damaged / damaged_total) * 100 if damaged_total > 0 else 0
    ok_accuracy = (tp_ok / ok_total) * 100 if ok_total > 0 else 0
    overall_accuracy = (tp_damaged + tp_ok) / total_samples * 100

    damaged_precision = (tp_damaged / (tp_damaged + fp_damaged)) * 100 if (tp_damaged + fp_damaged) > 0 else 0
    ok_precision = (tp_ok / (tp_ok + fp_ok)) * 100 if (tp_ok + fp_ok) > 0 else 0

    scores = np.max(predictions, axis=1)

    is_correct = [true == pred for true, pred in zip(true_classes, predict_index)]

    evaluation_results = {
        'correct_categories': [lb.classes_[i] for i in true_classes],
        'predicted_categories': [lb.classes_[i] for i in predict_index],
        'is_correct': is_correct,
        'predictions': [float(pred) for pred in scores],
        'scores': [float(score) for score in scores],
        'execution_time': prediction_times, 
        'total_damaged': tp_damaged,
        'total_ok': tp_ok,
        'accuracy_damaged': damaged_accuracy,
        'accuracy_ok': ok_accuracy,
        'precision_damaged': damaged_precision,
        'precision_ok': ok_precision,
        'overall_accuracy': overall_accuracy,
        'image_paths': file_paths,
        'batch_size': batch_size
    }

    # Print detailed statistics
    print(f"\nClassification Results:")
    print(f"Total test samples: {total_samples}\n")
    print(f"Total 'Damaged' test samples: {damaged_total}")
    print(f"Total 'Ok' test samples: {ok_total}")
    print(f"Accuracy for 'Damaged': {damaged_accuracy:.2f}% ({tp_damaged} out of {damaged_total})")
    print(f"Accuracy for 'Ok': {ok_accuracy:.2f}% ({tp_ok} out of {ok_total})")
    print(f"Precision for 'Damaged': {damaged_precision:.2f}% ({tp_damaged} out of {tp_damaged + fp_damaged})")
    print(f"Precision for 'Ok': {ok_precision:.2f}% ({tp_ok} out of {tp_ok + fp_ok})")
    print(f"Overall accuracy: {overall_accuracy:.2f}%")
    
    return evaluation_results


