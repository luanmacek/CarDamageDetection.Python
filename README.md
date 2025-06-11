# CarStudent Deep Learning Model

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.x-orange.svg)

CarStudent is a deep learning project for training and testing a convolutional neural network (CNN) for image classification. It uses TensorFlow and Keras Tuner for model training, supports both GPU and CPU, and integrates with a SQL Server database to store training and evaluation results.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Database Setup](#database-setup)
- [Notes](#notes)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features
- Train a CNN with hyperparameter tuning (learning rate, units) using Keras Tuner.
- Evaluate pre-trained models on test datasets with batch processing.
- Support for GPU and CPU training/evaluation.
- Organized image dataset handling (training, validation, test).
- SQL Server integration for storing training metadata and evaluation results.
- Early stopping to prevent overfitting.

## Requirements
- Python 3.8+
- TensorFlow 2.x
- Keras Tuner
- PyODBC
- Scikit-learn
- SQL Server with ODBC Driver 17
- NVIDIA GPU (optional, for GPU training)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/carstudent.git
   cd carstudent
   ```

2. Install dependencies:
   ```bash
   pip install tensorflow keras-tuner pyodbc scikit-learn
   ```

3. Set up SQL Server:
   - Ensure a SQL Server instance is running (default: `172.16.4.222,1433`).
   - Update the connection string in `main.py` if needed:
     ```python
     conn_str = 'DRIVER={ODBC Driver 17 for SQL Server};SERVER=your_server;DATABASE=CarStudent;UID=your_username;PWD=your_password;'
     ```
   - Install [ODBC Driver 17 for SQL Server](https://docs.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server).

4. Prepare the dataset:
   - Organize images into:
     ```
     images/
     ├── training/
     ├── validation/
     ├── test/
     ```
   - Ensure images are categorized by class labels.

## Usage
Run the script in either `train` or `test` mode using command-line arguments.

### Training a Model
Train a new model:
```bash
python main.py --mode train --device gpu
```
- `--mode train`: Activates training mode.
- `--device gpu`: Uses GPU (use `cpu` for CPU).
- The script will:
  - Load and preprocess images from `images/training/` and `images/validation/`.
  - Tune hyperparameters using Keras Tuner.
  - Train the model with early stopping.
  - Save the model to `models/YYYY-MM-DD HH-MM-SS/` and log results to the database.

### Testing a Model
Evaluate a pre-trained model:
```bash
python main.py --mode test --device gpu --batch_size 64 --date "YYYY-MM-DD HH-MM-SS"
```
- `--mode test`: Activates testing mode.
- `--device gpu`: Uses GPU.
- `--batch_size`: Sets batch size (default: 64).
- `--date`: Specifies model timestamp (optional; defaults to latest).
- The script will:
  - Load the model from `models/`.
  - Evaluate on `images/test/`.
  - Save results to the database.

## Project Structure
```
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
```
- `utils.py`: Data preprocessing, database initialization, and result-saving functions.
- `model.py`: CNN model architecture and evaluation logic.
- `main.py`: Main script for training/testing.
- `images/`: Dataset directories.
- `models/`: Saved models and data.

## Database Setup
The `CarStudent` database stores training metadata and evaluation results. Initialize tables by running:
```python
initialize_database(conn)
```
Ensure the SQL Server connection is configured correctly.

## Notes
- GPU training requires CUDA and cuDNN.
- Training uses a batch size of 16, 50 epochs, and early stopping (patience: 5).
- Testing defaults to a batch size of 64.
- Models are saved with timestamps for versioning.

## Contributing
Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit changes (`git commit -m 'Add feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.

## License
[MIT License](LICENSE)

## Contact
For questions or support, email [luandeveloper@gmail.com](mailto: luandeveloper1@gmail.com).
