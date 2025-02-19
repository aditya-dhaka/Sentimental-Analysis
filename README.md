# Sentimental-Analysis

This project demonstrates a comprehensive sentiment analysis pipeline for restaurant reviews using machine learning. The primary objective is to classify reviews as either positive or negative based on the textual content. This is achieved through the application of various natural language processing (NLP) techniques and the evaluation of multiple machine learning classifiers. The pipeline includes preprocessing, feature extraction, model training, and performance evaluation, using both traditional machine learning algorithms and advanced techniques.

Technologies and Libraries used:

1. Python 3.x: Programming language for data processing and model building.
2. pandas: Data manipulation and analysis.
3. nltk: Natural Language Processing (NLP) tasks, such as text cleaning, tokenization, stopword removal, and stemming.
4. scikit-learn: Machine learning models and utilities for evaluation, including Naive Bayes, Random Forest, and train-test splitting.
5. XGBoost: A powerful gradient boosting algorithm often used for classification tasks.
6. matplotlib (optional): For visualization of results (not included in the code but can be added).

Dataset

The dataset consists of restaurant reviews in a tab-separated format (Restaurant Reviews.tsv). Each entry includes the review text and its corresponding sentiment label (positive or negative). The dataset is processed for sentiment classification tasks.

How It Works

Data Loading:
• The dataset is loaded using pandas.read_csv, which reads the Restaurant Reviews.tsv file. 
• The first few rows of the dataset are previewed to ensure correct data structure.

Text Preprocessing:
• Cleaning: Non-alphabetical characters are removed using regular expressions. 
• Normalization: The text is converted to lowercase to ensure uniformity. 
• Tokenization: Reviews are split into individual words. 
• Stopword Removal: Common stopwords (e.g., "the", "and", "is") are removed using NLTK’s stopword list. 
• Stemming: Words are reduced to their root form using PorterStemmer.

Feature Extraction:
• The text data is transformed into numerical feature vectors using CountVectorizer, which counts word occurrences. 
• Only the top 800 most frequent words are used as features to reduce dimensionality.

Model Training:
• The dataset is split into training and testing sets (80% training, 20% testing). 
• The following classifiers are trained: 
• Gaussian Naive Bayes (GNB): Assumes that features follow a Gaussian distribution. 
• Multinomial Naive Bayes (MNB): Suitable for text classification tasks where features are word counts. 
• Bernoulli Naive Bayes (BNB): Works well with binary/boolean features. 
• Random Forest Classifier: An ensemble method that builds multiple decision trees to improve accuracy. 
• XGBoost Classifier: A high-performance gradient boosting algorithm that typically outperforms other models in classification tasks.

Model Evaluation:
• The performance of each model is evaluated using accuracy scores. 
• The best-performing model is identified based on accuracy. 
• The confusion matrix of the best model is displayed to evaluate its classification performance in more detail.

Results

The results include the accuracy scores for all classifiers, with the XGBoost model generally performing the best. The confusion matrix for the best model is also printed for a deeper understanding of its predictions.

Here are some sample of result

![Screenshot 2025-01-19 152942](https://github.com/user-attachments/assets/69569c99-6493-4b03-a817-fbcfa319dae1)
![Screenshot 2025-01-19 153100](https://github.com/user-attachments/assets/8b30b8b4-4dad-4cb3-9be2-e775138c6daf)

Contributing

We welcome contributions from the community. If you would like to contribute, please follow the steps below:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and test them.
4. Submit a pull request with a detailed explanation of your changes.
5. Please ensure that your code adheres to the style guide and includes relevant tests.

