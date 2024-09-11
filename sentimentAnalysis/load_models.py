import pickle

# Load the TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Load the Logistic Regression model
with open('LogisticRegression.pkl', 'rb') as file:
    LogisticRegression = pickle.load(file)

# Load the Linear SVC model
with open('LinearSVC.pkl', 'rb') as file:
    LinearSVC = pickle.load(file)

# Load the Multinomial NaiveBayes model
with open('MultinomialNaiveBayes.pkl', 'rb') as file:
    MultinomialNaiveBayes = pickle.load(file)

# Load the Random Forest model
with open('RandomForest.pkl', 'rb') as file:
    RandomForest = pickle.load(file)

# Load the Decision Tree model
with open('DecisionTree.pkl', 'rb') as file:
    DecisionTree = pickle.load(file)

