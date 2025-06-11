CarStudent Deep Learning Model
Overview
CarStudent is a deep learning project designed for training and testing a convolutional neural network (CNN) for image classification tasks. The project uses TensorFlow and Keras Tuner for model training, with support for both GPU and CPU devices. It integrates with a SQL Server database to store training and evaluation results. The project is structured to handle image preprocessing, model training with hyperparameter tuning, and evaluation, with data organized into training, validation, and test sets.
Features

Train Mode: Trains a CNN model using Keras Tuner for hyperparameter optimization (learning rate, number of units).
Test Mode: Evaluates a pre-trained model on a test dataset and saves results to a SQL Server database.
Device Support: Configurable to run on GPU or CPU.
Data Management: Organizes image datasets into training, validation, and test directories.
Database Integration: Stores training metadata and evaluation results in a SQL Server database.
Early Stopping: Implements early stopping to prevent overfitting during training.

Requirements

Python 3.8+
TensorFlow 2.x
Keras Tuner
PyODBC
Scikit-learn
SQL Server (or compatible ODBC driver)
NVIDIA GPU (optional, for GPU training)

Installation

Clone the Repository:
git clone https://github.com/your-username/carstudent.git
cd carstudent


Install Dependencies:
pip install tensorflow keras-tuner pyodbc scikit-learn


Set Up SQL Server:

Ensure you have a SQL Server instance running (e.g., at 172.16.4.222,1433).
Update the connection string in the script if necessary:conn_str = 'DRIVER={ODBC Driver 17 for SQL Server};SERVER=your_server;DATABASE=CarStudent;UID=your_username;PWD=your_password;'


Install the ODBC Driver 17 for SQL Server if not already installed.


Prepare Data:

Organize your image dataset into the following structure:images/
├── training/
├── validation/
├── test/


Ensure images are placed in the respective directories with appropriate class labels.



Usage
The script supports two modes: train and test. Use the command-line arguments to configure the script.
Training a Model
To train a new model:
python main.py --mode train --device gpu


--mode train: Runs the script in training mode.
--device gpu: Uses GPU for training (use cpu for CPU training).
The script will:
Load and preprocess images from images/training/ and images/validation/.
Perform hyperparameter tuning using Keras Tuner.
Train the model with early stopping.
Save the model to models/YYYY-MM-DD HH-MM-SS/ and log results to the database.



Testing a Model
To evaluate a pre-trained model:
python main.py --mode test --device gpu --batch_size 64 --date "YYYY-MM-DD HH-MM-SS"


--mode test: Runs the script in testing mode.
--device gpu: Uses GPU for evaluation.
--batch_size: Sets the batch size for testing (default: 64).
--date: Specifies the model date to load (format: YYYY-MM-DD HH-MM-SS). If omitted, the latest model is used.
The script will:
Load the specified (or latest) model from models/.
Evaluate it on the test dataset in images/test/.
Save evaluation results to the SQL Server database.



Project Structure
carstudent/
├── images/
│   ├── training/
│   ├── validation/
│   ├── test/
├── models/
│   ├── YYYY-MM-DD HH-MM-SS/
│   │   ├── model_YYYY-MM-DD HH-MM-SS.keras
│   │   ├── training/
│   │   ├── validation/
│   │   ├── test/
├── utils.py
├── model.py
├── main.py
└── README.md


utils.py: Contains helper functions for data preprocessing, database initialization, and result saving.
model.py: Defines the CNN model architecture and evaluation logic.
main.py: Main script for training and testing the model.
images/: Directory for training, validation, and test image datasets.
models/: Directory where trained models and their associated data are saved.

Database Schema
The script assumes a SQL Server database (CarStudent) with tables to store:

Training metadata (e.g., training date, accuracy, model path).
Evaluation results (e.g., model performance metrics).

Run initialize_database(conn) to set up the necessary tables.
Notes

Ensure the GPU is properly configured with CUDA and cuDNN for GPU training.
The script uses a batch size of 16 for training and 64 for testing by default.
Training uses early stopping with a patience of 5 epochs to monitor validation loss.
Models are saved with a timestamp (YYYY-MM-DD HH-MM-SS) for easy identification.

Contributing
Contributions are welcome! Please submit a pull request or open an issue for bug reports, feature requests, or improvements.
License
This project is licensed under the MIT License. See the LICENSE file for details.
Contact
For questions or support, contact your-email@example.com.
