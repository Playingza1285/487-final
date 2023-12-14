# EECS 487 Final Project - Bias Detecton Within Mass Media

Group Members: Andrew Shin, William Chen, Vincent Weng

## Problem Description

In an era where the media plays a pivotal role in shaping public opinion, identifying bias has become increasingly important. For much of the public, the media is the main, if not only, source of information and influences the public’s perception of most issues or ideas. As a result, our team has focused this project on recognizing the political standings of the authors and discerning the underlying biases in articles. 
If successful, this initiative aims to promote transparency in news outlets and provide insight to the public on the influence the media has on their opinions.

Our team investigates author bias by analyzing the language elements like word choice or syntax in the input text data and then predicting the author’s ideological positions as the main output. 
This can be a vital tool for detecting bias within political spheres and discerning biased opinions within articles overall.

## Dataset
For our purposes, our team utilized a dataset containing information on 37,554 articles from www.allsides.com. These articles are collected and converted to JSON formats from this public github repository: https://github.com/ramybaly/Article-Bias-Prediction.

Example JSON Format:
```
{
    "topic": "terrorism",
    "source": "New York Times - News",
    "bias": 0,
    "url": 
    "http://www.nytimes.com/”
    "title": "Bomb Suspect Changed After Trip Abroad, Friends Say",
    "date": "2016-09-20",
    "authors": "N. R. Kleinfield",
    "content_original": "Besides his most recent trip to      Quetta, Mr. Rahami visited Karachi, Pakistan, in 2005. Both of those cities\u2019 reputations have become entwined with the militant groups…”
    "source_url": "www.nytimes.com",
    "bias_text": "left",
    "ID": "004Gt3gcsotuiYmz"
}
```

## Methodology

### Dataset Overview:

* The dataset comprises 37,554 articles sourced from www.allsides.com, credited to the GitHub repository of Ramy Baly.
* The dataset is provided in JSON format with attributes such as ID, topic, source, URL, date, authors, title, content_original, content, bias_text, and bias.

### Data Cleaning and Selection:

* Checked for empty attributes and filtered features for preprocessing, including ID, content_original, bias_text, and bias.
* Selected features crucial for supervised machine learning, ensuring unique identification, textual analysis, and bias labeling.
* Addressed potential utility of additional attributes like topic, date, and title for future analysis.

### Dataset Suitability and Class Distribution:

* The dataset, with bias classes 0, 1, and 2 representing left, center, and right affiliations, provides labeled data for supervised learning.
* Classes are distributed as follows: class 0 (34.6%), class 1 (28.8%), and class 2 (36.6%), showing relatively balanced representation.
* The dataset's richness enables machine learning techniques for identifying and understanding biases across the political spectrum.

### Data Preprocessing:

* Employed basic preprocessing techniques, including tokenization, stop word removal, lemmatization, and stemming for efficient text analysis.
* Transformed the processed dataset into a Pandas DataFrame for ease of manipulation, resulting in a CSV file for portability.

### Multinomial Naive Bayes and DistilBERT:

* Adopted a combined approach using Multinomial Naive Bayes and DistilBERT for detecting political bias.
* Multinomial Naive Bayes established a baseline accuracy by analyzing term frequencies, while DistilBERT provided advanced analysis using transformer-based neural networks.

### Multinomial Naive Bayes Architecture:

The dataset is converted into a Pandas DataFrame, split into 80% training and 20% testing sets. The feature (article content) and target variables (political bias) are defined. The pipeline involves CountVectorizer() for text-to-numerical conversion and MultinomialNB() for classification. The architecture seamlessly integrates feature extraction and classification within a scikit-learn pipeline, streamlining text classification and bias detection.

### DistilBERT Architecture:
Before using DistilBERT, punctuation is removed, and content is tokenized, including [CLS] and [SEP] tokens, using DistilBERT's own tokenizer. Subword tokenization is applied, replacing unseen tokens with [UNK]. Truncation is necessary due to the 512-feature limit, initially attempted but later reduced to 150 features to improve model training efficiency. The model structure involves sending vectorized content through DistilBERT, followed by a linear layer for dimension reduction and a softmax layer for bias determination using the [CLS] token. Training involves cross-entropy loss and the Adam optimizer.

The dataset is split into training, validation, and test sets (80/10/10). A hyperparameter search yields optimal learning rate (0.0001) and weight decay (0.0001). The batch size is 64, and each model undergoes 25 epochs. Evaluation metrics include accuracy and a confusion matrix to assess bias prediction. The Multinomial Naive Bayes classifier achieves a baseline accuracy of 0.64, considering class imbalances, particularly for class 1.

### DistilBERT Model Training:

* Preprocessed data for DistilBERT involved punctuation removal, tokenization, subword tokenization using DistilBERT's tokenizer, and content truncation.
* Truncated content to 150 tokens due to DistilBERT's feature limit, optimizing for efficient training.
* Detailed the DistilBERT model structure, involving linear layers, softmax activation, and cross-entropy loss during training.

