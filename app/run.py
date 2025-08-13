import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db') 
df = pd.read_sql_table('DisasterResponse_table', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # --- VISUALIZATION 1: Distribution of Message Genres (Original visual) ---
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # --- EXTRACT DATA FOR NEW VISUALIZATIONS ---
    
    # Visualization 2: Top 10 Categories
    # We sum the values for each category column (all columns after the first 4)
    # and sort them to find the most frequent ones.
    category_counts = df.iloc[:,4:].sum().sort_values(ascending=False)
    top_10_category_names = list(category_counts.index[:10])
    top_10_category_counts = list(category_counts.values[:10])

    # Visualization 3: Category distribution by genre 'direct' vs. 'news'
    # We filter the dataframe by genre and then sum the categories for each.
    direct_category_counts = df[df['genre'] == 'direct'].iloc[:,4:].sum().sort_values(ascending=False)
    news_category_counts = df[df['genre'] == 'news'].iloc[:,4:].sum().sort_values(ascending=False)
    # For consistency, we use the same categories as in the top-10 chart.
    direct_genre_counts_top10 = direct_category_counts[top_10_category_names]
    news_genre_counts_top10 = news_category_counts[top_10_category_names]


    # --- CREATE VISUALS ---
    graphs = [
        # Graph 1: Distribution of Message Genres
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        # Graph 2: Top 10 Disaster Categories
        {
            'data': [
                Bar(
                    x=top_10_category_names,
                    y=top_10_category_counts
                )
            ],
            'layout': {
                'title': 'Top 10 Message Categories',
                'yaxis': {
                    'title': "Number of Messages"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': -35  # Rotates labels for better readability
                }
            }
        },
        # Graph 3: Category Distribution by Genre
        {
            'data': [
                Bar(
                    x=top_10_category_names,
                    y=direct_genre_counts_top10,
                    name='Direct Message' # Legend entry
                ),
                Bar(
                    x=top_10_category_names,
                    y=news_genre_counts_top10,
                    name='News Article' # Legend entry
                )
            ],
            'layout': {
                'title': 'Category Distribution by Genre (Top 10)',
                'yaxis': {
                    'title': "Number of Messages"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': -35
                },
                'barmode': 'stack' # Stacks the bars on top of each other
            }
        }
    ]
    
    # Encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)
# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
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