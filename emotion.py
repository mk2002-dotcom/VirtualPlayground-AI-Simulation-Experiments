# emotion
from transformers import pipeline

sentiment = pipeline("sentiment-analysis")

print(sentiment("I really love this movie!"))
print(sentiment("This is the worst thing ever."))