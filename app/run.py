import json
import plotly
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Heatmap
import joblib
from sqlalchemy import create_engine

from pipeline.features import LengthExtractor

app = Flask(__name__)

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('messages_categorized', engine)

# load model
model = joblib.load("models/MOC_forest.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    categories_names = df.columns[3:]
    categories_counts = df[categories_names].sum().sort_values(ascending=False)

    # Calculate correlation between the labels
    correlation_values = df[categories_names].corr().values.tolist()

    # Calculate the Euclidean distance between the category vectors; the lower the value, the higher the similariy
    categories_df = df[categories_names]
    eucl_dist_values = np.sqrt(categories_df.T.dot(categories_df == 0) + (categories_df == 0).T.dot(categories_df)).values.tolist()
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                    marker_color='#341C4E'
                )
            ],
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                },
                "margin": {
                    "b": 120
                }
            }
        }, 
        {
            "data": [
                Bar(
                    x=categories_counts.index.tolist(),
                    y=categories_counts,
                    marker_color='#341C4E'
                )
            ],
            "layout": {
                "title": "Distribution of Message Categories",
                "yaxis": {
                    "title": "Count"
                },
                "xaxis": {
                    "title": "Category",
                    "tickangle": 45
                },
                "margin": {
                    "b": 120
                } 
            }
        }, 
        {
            "data": [
                Heatmap(
                    z=correlation_values,
                    x=categories_names,
                    y=categories_names,
                    colorscale='Viridis'
                ),
            ],
            "layout": {
                "title": "Paiwise Correlation between Categories",
                "margin": {
                    "b": 120,
                    "l": 100,
                    "t": 50,
                    "r": 0
                },
                "xaxis": {
                    "tickangle": 45
                },
                "yaxis":{
                    "tickangle": -45
                }
            }                
        }, 
        {
            "data": [
                Heatmap(
                    z=eucl_dist_values,
                    x=categories_names,
                    y=categories_names,
                    colorscale='Viridis',
                    reversescale=True
                ),
            ],
            "layout": {
                "title": "Pairwise Euclidean Distance between Categories",
                "margin": {
                    "b": 120,
                    "l": 100,
                    "t": 50,
                    "r": 0
                },
                "xaxis": {
                    "tickangle": 45
                },
                "yaxis":{
                    "tickangle": -45
                }
            }                
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    print(classification_labels)
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()