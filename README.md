# Disinformation-Detection-in-News-Articles-Using-Machine-Learning
This project explores using   data mining and machine learning to classify news articles as either fake or real. It addresses the challenge of disinformation, which is deliberately false information spread to deceive, manipulate, or cause public harm.

## Dataset and Preprocessing
The foundation for this project was a dataset obtained from Kaggle.

The original dataset contained 2,096 news articles with 12 columns, including fields like author, title, text, site_url, and label.
The articles were multilingual, with 2,017 in English, 72 in German, 2 in French, and 1 in Spanish.

A series of cleaning steps were performed to prepare the data for analysis:
All non-English articles were removed.
Four columns deemed unnecessary for the experiments (site_url, main_img_url, type, hasImage) were deleted.
Rows with missing values in critical fields, such as the article text, were removed.

After cleaning, the dataset was reduced to 1,966 articles and 8 columns.

## Feature Engineering and Modeling

To convert text into a numerical format, Term Frequency-Inverse Document Frequency (TF-IDF) was used. This technique assigns importance to words based on their frequency in an article relative to their frequency across all articles. The following TF-IDF configurations were tested:


* Minimum Document Frequency (min_df): 5, 10, and 15.
* N-gram Ranges: (1,1) for unigrams, (1,2) for bigrams, and (1,3) for trigrams.
* Maximum Features: 1000, 2000, and 3000.


The dataset was imbalanced, with 62% of articles labeled as fake and 38% as real. To prevent model bias towards the majority class, the Synthetic Minority Oversampling Technique (SMOTE) was applied during the training phase. SMOTE works by generating new, synthetic examples of the minority class (real news) rather than simply duplicating existing ones.



Three supervised learning models were used for classification:

* Decision Tree: A flowchart-like model that makes decisions based on the presence of certain words.
* Multinomial Naive Bayes: A probabilistic model that estimates the likelihood an article is fake based on word frequencies.
* k-Nearest Neighbors (k-NN): A model that classifies articles by comparing them to the most similar articles in the training set.

## Experimental Design

Two separate experiments were conducted to evaluate model performance on different parts of the articles.

* Experiment 1 (Full Text): Models were trained and tested using the text_without_stopwords column, which contains the main body of each article.
* Experiment 2 (Headlines): Models were trained and tested using the title_without_stopwords column to see if headlines alone were sufficient for classification.

For each experiment, the models were evaluated across three different train-test split ratios: 

* 80/20
* 70/30
* 60/40

The F1 score was the primary metric used for evaluation, as it provides a balanced measure of precision and recall, which is crucial for imbalanced datasets.

## Conclusion

Multinomial Naive Bayes proved to be the most effective model for classifying full articles, while Decision Trees excelled when analyzing headlines alone. In contrast, k-Nearest Neighbors consistently struggled across both experiments due to a high false positive rate. The project also highlighted that simpler TF-IDF parameters, such as single-word tokenization and smaller feature sets, tended to perform better.

## Future Work
This study has some limitations, including a relatively small dataset and the use of static train-test splits without cross-validation. 

Future work could explore:

* Larger and more diverse datasets.
* Implementing robust evaluation techniques like k-fold cross-validation.
* Experimenting with different models such as Random Forest, AdaBoost, or transformer-based approaches like BERT.
* Using word embedding techniques like Word2Vec or GloVe to capture semantic relationships between words.

