# disaster_response_pipeline



`!pip install iterative-stratification`

The README file includes a summary of the project, how to run the Python scripts and web app, and an explanation of the files in the repository. 
Comments are used effectively and each function has a docstring.


# Disaster Response Pipeline

This is an example file with default selections.

## Content


## Install


If you are using `anaconda`/`miniconda` use following command to install all required libraries:
```
conda create --name <env> --file requirements.txt
```
The project makes use of the library `iterative-stratification`


## Usage

The following commands make use of the `-m` flag when running the scripts. This is done to avoid issues with 
accessing higher-level modules from sub-directories. The official documentation for this option can be found [here](https://docs.python.org/3/using/cmdline.html).

To clean the data, run the following command in the directory `disaster_response_pipeline`:
```
python3 -m data.process_data data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```

To re-run the model building, change directory to `disaster_response_pipeline` and run:
```
python3 -m models.train_classifier data/DisasterResponse.db my_model_name
```
This will train a classifier based on the data in the provided database and save the model to the `models` directory.

To start the web app, run
```
python3 -m  app.run
```
in the root directory of the project folder `disaster_response_pipeline` .


## Contributing

PRs accepted.

## License

MIT © A. Q.