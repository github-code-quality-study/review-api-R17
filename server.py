import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

# Download NLTK data
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

# Set up the sentiment analyzer and stop words
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

# Read reviews from CSV into a list of dictionaries
reviews = pd.read_csv('data/reviews.csv').to_dict('records')

class ReviewAnalyzerServer:
    # Initialize the class with any necessary attributes
    def __init__(self) -> None:
        pass

    # Analyze sentiment of a given review body
    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    # Handle incoming requests
    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        # Determine the request method
        if environ["REQUEST_METHOD"] == "GET":
            return self.handle_get_request(environ, start_response)

        if environ["REQUEST_METHOD"] == "POST":
            return self.handle_post_request(environ, start_response)

    def handle_get_request(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        # Parse the query string
        query = parse_qs(environ.get('QUERY_STRING', ''))
        location = query.get('location', [None])[0]
        start_date = query.get('start_date', [None])[0]
        end_date = query.get('end_date', [None])[0]

        # Filter reviews based on query parameters
        filtered_reviews = reviews

        if location:
            filtered_reviews = [review for review in filtered_reviews if review['Location'] == location]
        
        if start_date:
            filtered_reviews = [review for review in filtered_reviews if datetime.strptime(review['Timestamp'], '%Y-%m-%d %H:%M:%S') >= datetime.strptime(start_date, '%Y-%m-%d')]
        
        if end_date:
            filtered_reviews = [review for review in filtered_reviews if datetime.strptime(review['Timestamp'], '%Y-%m-%d %H:%M:%S') <= datetime.strptime(end_date, '%Y-%m-%d')]

        # Analyze sentiment for each review
        for review in filtered_reviews:
            review['sentiment'] = self.analyze_sentiment(review['ReviewBody'])

        # Sort reviews by descending compound sentiment score
        filtered_reviews.sort(key=lambda r: r['sentiment']['compound'], reverse=True)

        # Create the response body as a JSON byte string
        response_body = json.dumps(filtered_reviews, indent=2).encode('utf-8')

        # Set the appropriate response headers
        start_response("200 OK", [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(response_body)))
        ])

        return [response_body]

    def handle_post_request(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        try:
            content_length = int(environ.get('CONTENT_LENGTH', 0))
        except (ValueError):
            content_length = 0

        request_body = environ['wsgi.input'].read(content_length).decode('utf-8')
        post_data = parse_qs(request_body)

        review_body = post_data.get('ReviewBody', [None])[0]
        location = post_data.get('Location', [None])[0]

        # Define allowed locations
        allowed_locations = [
            'Albuquerque, New Mexico', 'Carlsbad, California', 'Chula Vista, California',
            'Colorado Springs, Colorado', 'Denver, Colorado', 'El Cajon, California',
            'El Paso, Texas', 'Escondido, California', 'Fresno, California', 'La Mesa, California',
            'Las Vegas, Nevada', 'Los Angeles, California', 'Oceanside, California',
            'Phoenix, Arizona', 'Sacramento, California', 'Salt Lake City, Utah', 'San Diego, California',
            'Tucson, Arizona'
        ]

        # Validate review body and location
        if not review_body or not location:
            start_response("400 Bad Request", [("Content-Type", "text/plain")])
            return [b"Both ReviewBody and Location are required fields."]

        if location not in allowed_locations:
            start_response("400 Bad Request", [("Content-Type", "text/plain")])
            return [b"Invalid location provided."]

        review_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        new_review = {
            'ReviewId': review_id,
            'Location': location,
            'Timestamp': timestamp,
            'ReviewBody': review_body,
            'sentiment': self.analyze_sentiment(review_body)
        }

        reviews.append(new_review)

        response_body = json.dumps(new_review, indent=2).encode('utf-8')

        start_response("201 Created", [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(response_body)))
        ])

        return [response_body]


if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = int(os.environ.get('PORT', 8000))
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()
