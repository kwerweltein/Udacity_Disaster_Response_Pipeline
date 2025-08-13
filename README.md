# Disaster Response Pipeline Project

T. Bender for udacity


## About The Project

This Project is part of Data Science Nanodegree Program by Udacity in collaboration with Figure Eight. The dataset contains pre-labelled tweet and messages from real-life disaster events. The project aim is to build a Natural Language Processing (NLP) model to categorize messages on a real time basis.

This project is divided in the following key sections:

1. Processing data, clean the data and save them in a SQLite DB
2. Build a machine learning pipeline
3. Run a web app which can show model results in real time


## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Go to: http://127.0.0.1:3001

## Imported Libraries & Modules

This project leverages a comprehensive set of Python libraries and modules for various tasks, including data processing, machine learning model building, and natural language processing. Key functionalities provided by these imports include:

*   **Core Data Handling:** `sys`, `os`, `re`, `numpy` (for numerical operations), and `pandas` (for data manipulation and analysis).
*   **Database Interaction:** `sqlalchemy` is used to create database engines, facilitating data loading and saving.
*   **Serialization:** `pickle` allows for saving and loading Python objects, such as trained models.
*   **Statistical Operations:** `scipy.stats.gmean` is imported for geometric mean calculations.
*   **Scikit-learn (Machine Learning):** A wide array of modules from `sklearn` are utilized for building and evaluating machine learning pipelines:
    *   `Pipeline`, `FeatureUnion` for constructing complex processing workflows.
    *   `train_test_split` for splitting data into training and testing sets.
    *   `classification_report`, `confusion_matrix`, `fbeta_score`, `make_scorer` for model evaluation.
    *   `GradientBoostingClassifier`, `RandomForestClassifier`, `AdaBoostClassifier` as core classification algorithms.
    *   `GridSearchCV` for hyperparameter tuning.
    *   `TfidfTransformer`, `CountVectorizer` for text feature extraction.
    *   `MultiOutputClassifier` for handling multi-label classification problems.
    *   `BaseEstimator`, `TransformerMixin` for creating custom scikit-learn compatible components.
*   **Natural Language Toolkit (NLTK):** `nltk` is employed for text preprocessing tasks, including:
    *   `word_tokenize` for tokenizing text into words.
    *   `WordNetLemmatizer` for lemmatizing words.


## Repository Structure
```
├── app/ : Folder contains the webapp
    ├── templates/
    └── run.py : the webapp
├── data/ : Folder for data cleaning
    ├── categories.csv : 
    ├── messages.csv
    ├── DisasterResponse.db 
    ├── ETL Pipeline Preparation.ipynb : development notebook
    └── process_data.py : final deployment data processing
├── models/ : Folder contains the ML training
    ├── classifier.pkl : output of train_classifier.py
    ├── ML Pipeline Preparation.ipynb  : development notebook
    └── train_classifier.py : final deployment ML training
└── README.md
```


## Acknowledgments
* **Figure Eight** for providing the relevant dataset to train the model
*   **ASK Bosch** The BOSCH LLM for useful debugging help.