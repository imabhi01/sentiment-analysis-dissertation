from django.shortcuts import render
from django.http import JsonResponse
from .load_models import vectorizer, LinearSVC, LogisticRegression, MultinomialNaiveBayes, RandomForest, DecisionTree
import pickle
from .utils import predict_sentiment

def index(request):
    return render(request, 'index.html')

def analyze_sentiment(reviewText):

    # Vectorizing the text for prediction
    review_vector = vectorizer.transform([reviewText])

    # Predict using different models
    linearSVC = LinearSVC.predict(review_vector)[0]
    logisticReg = LogisticRegression.predict(review_vector)[0]
    multinomialNaiveBayes = MultinomialNaiveBayes.predict(review_vector)[0]
    randomForest = RandomForest.predict(review_vector)[0]
    decisionTree = DecisionTree.predict(review_vector)[0]

    return {
        'Linear_SVC': linearSVC,
        'Logistic_Regression': logisticReg,
        'Multinomial_NaiveBayes': multinomialNaiveBayes,
        'Random_Forest': randomForest,
        'Decision_Tree': decisionTree,
    }


def analyze(request):
    if request.method == 'POST':
        reviewText = request.POST['reviewText']
        sentiments = analyze_sentiment(reviewText)
        sentiment = predict_sentiment(reviewText)
        if(sentiment == 2):
            mood = 'Positive'
        elif(sentiment == 1):
            mood = 'Neutral'
        else:
            mood = 'Negative'
    return render(request, 'result.html', {'sentiments': sentiments, 'mood': mood})

def predict(request):
    if request.method == 'POST':
        reviewText = request.POST.get('reviewText')
        sentiment = predict_sentiment(reviewText)
        if(sentiment == 2):
            mood = 'Positive'
        elif(sentiment == 1):
            mood = 'Neutral'
        else:
            mood = 'Negative'
        return JsonResponse({'mood': mood})
    
    return render(request, 'index.html')
        
    