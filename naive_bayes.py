import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

df = pd.read_csv("data.csv")

random_sample = df.sample(n=5000, random_state=42)

random_sample.to_csv("random_sample.csv", index=False)

train_data, test_data, train_bias, test_bias = train_test_split(
    df['content_original'],
    df['bias_text'],
    test_size=0.2,
    random_state=42
)

bias_pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

# Prints counts of samples in each class in the training set
print("Training Set Class Distribution:")
print(train_bias.value_counts())

# Prints counts of samples in each class in the test set
print("\nTest Set Class Distribution:")
print(test_bias.value_counts())

# Trains Bias Classifier
bias_pipeline.fit(train_data, train_bias)

# Makes predictions
bias_predictions = bias_pipeline.predict(test_data)

# Evaluates the Bias Classifier
bias_accuracy = accuracy_score(test_bias, bias_predictions)
print(f"Bias Accuracy: {bias_accuracy:.2f}")

# Additional evaluation metrics
bias_precision, bias_recall, bias_f1, _ = precision_recall_fscore_support(test_bias, bias_predictions, average='weighted')
print(f"Bias Precision: {bias_precision:.2f}")
print(f"Bias Recall: {bias_recall:.2f}")
print(f"Bias F1 Score: {bias_f1:.2f}")

print("Bias Classification Report:")
print(classification_report(test_bias, bias_predictions))