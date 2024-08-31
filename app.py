from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
from textblob import TextBlob

app = Flask(__name__)

# Load the vectorizer and model
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('XGBoost1.pkl', 'rb') as f:
    clf = pickle.load(f)

# Create a form for user input
class ReviewForm(Form):
    tweet = TextAreaField('Tweet', [validators.DataRequired(), validators.length(min=15)])

# Create a route for the homepage
@app.route('/')
def index():
    form = ReviewForm(request.form)
    return render_template('index.html', form=form)

# Create a route for the results page
@app.route('/results', methods=['POST'])
def results():
    form = ReviewForm(request.form)
    if request.method == 'POST' and form.validate():
        tweet = request.form['tweet']
        # Preprocess the tweet
        tweet = tweet.lower()
        # Vectorize the tweet
        tweet_tfidf = vectorizer.transform([tweet])
        # Make a prediction
        prediction = clf.predict(tweet_tfidf)[0]

        # Determine sentiment based on prediction
        sentiment = 'Positive' if prediction == 1 else 'Negative'
        
        # Additional sentiment analysis using TextBlob
        blob = TextBlob(tweet)
        polarity = blob.sentiment.polarity
        
        if polarity > 0:
            sentiment = 'Positive'
        elif polarity < 0:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'
        
        return render_template('results.html', content=tweet, sentiment=sentiment)
    return render_template('index.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)
