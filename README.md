# Disaster Response Pipeline

This project analyzes disaster data from Figure Eight to build a model for an API that classifies disaster messages. The API is used in a web app that accepts new messages and returns a classification into several categories. The web app also displays visualizations of the used data.

The focus of this implementation was not to obtain a perfect model but rather to provide a working protoype including preprocessing data, trainig a model, and using the model to make predictions for user input through a web app. 

## Content
The code of this project is structured in the following way: 
```
└── disaster_response_pipeline
    ├── app                            
    │   ├── __init__.py              # Initializes the app module
    │   ├── run.py                   # Runs the web app
    │   └── templates                
    │       ├── go.html              # Template for result page, extends master.html
    │       └── master.html          # Template for home page
    ├── data
    │   ├── __init__.py              # Initializes the data module
    │   ├── disaster_categories.csv  # Input dataset: Messages' categories 
    │   ├── disaster_messages.csv    # Input dataset: Messages' content 
    │   ├── DisasterResponse.db      # Database with merged dataset
    │   └── process_data.py          # Contains data preprocessing and saving to a database
    ├── models
    │   ├── __init__.py              # Initializes the models module
    │   ├── MOC_forest.pkl           # Trained model used for predictions
    │   └── train_classifier.py      # Contains steps to train and store a classifier model
    ├── pipeline
    │   ├── __init__.py              # Initializes the pipeline module
    │   ├── features.py              # Contains feature class definitions
    │   └── functions.py             # Contains functions to run the overall pipeline
    ├── LICENSE
    ├── README.md 
    └── requirements.txt 
```
    
## Install
All required packages are listed in `requirements.txt`. If you are using `anaconda`/`miniconda` use following command to set up a new environment 
with all requirements to run the code:
```
conda create --name <name of the environment> --file requirements.txt
```

## Usage
The following commands make use of the `-m` flag when running the scripts. This is used to avoid issues with accessing higher-level modules from sub-directories. The official documentation for this option can be found [here](https://docs.python.org/3/using/cmdline.html).

To clean the data, run the following command in the directory `disaster_response_pipeline`:
```
python3 -m data.process_data data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```

To re-run the model building, run
```
python3 -m models.train_classifier data/DisasterResponse.db my_model_name
```
from `disaster_response_pipeline`. This will train a classifier based on the data in the provided database and save the model to the `models` directory.

To start the web app, run
```
python3 -m  app.run
```
from `disaster_response_pipeline`.

## Label Imbalance
### Stratification
The labels in the provided dataset are highly imbalanced, e. g. the label `fire` only appears in 282 messages, while over 20,000 messages carry the 
label `related`. One issue that arises from this imbalance is a train-test-split with a label only present in one of the sets. To mitigate this 
issue, the project makes use of the `iterative-stratification` package by T. J. Bradberry, which provides scikit-learn compatible cross validators with stratification for multilabel data, contrary to the scikit-learn implementations which only support stratification for single label data. 

### Evaluation Metric
Another issue arising from the imbalance is the choice of an appropriate evaluation metric. The standard accuracy metric leads to meaningless results due to 
the imbalance, e.g. by never predicting a label that only exists in 1 % of all messages, an accuracy of 99 % can be achieved. However, for this use case we need to pay close attention to these messages since they contain relevant information about people in need and acting on those messages might save lives. Therefore, a more appropriate metric is one which reflects how many relevant messages of all relevant messages were classified correctly. 

A metric serving this purpose is 'recall'; it is defined as the fraction of the relevant items that are correctly classified of all relevant items [https://en.wikipedia.org/wiki/Precision_and_recall](Wikipedia: Precision and Recall). The flipside of this metric is that a model that only predicts the minority label obtains a perfect recall of `1`. A model like this would not be beneficial either since humans would still need to check all messages and evaluate them manually. 

To avoid selecting such a model, the 'F1-score' was chosen as the evaluation metric. This score also takes into account 'precision' which is defined as the fraction of correctly classified items of all items that were labeled with the respective label, regardless of the actual affiliation of a message to a label, i.e. it punishes a model that assigns a label to all messages. The F1-score is the harmonic mean of recall and precision [https://en.wikipedia.org/wiki/F1_score](Wikipedia: F1-score). 

A drawback of the standard F1-score is that it gives equal weight to recall and precision. By introducing an additional parameter more weight can be given to either precision or recall, however a metric might not be equally important for all categories. Therefore, one suggestion to improve the model selection is to use a custom evaluation scorer that assigns category-specific weights. For example, more weight could be given to recall in the case of the `fire` label and a relatively lower weight to the `electricity` label when determining the overall performance of a model, since missing messages containing the latter might have less severe consequences. In practice, the weighting should be done carefully taking into account potential consequences.

## Limitations of the Model
Due to limited computing resources, the grid search was performed only for few parameter combinations. A wider range of parameters is provided in the code
and can be uncommented to search more combinations.

## Contributing
PRs accepted, feel free to contribute.

## License
MIT © A. Q.