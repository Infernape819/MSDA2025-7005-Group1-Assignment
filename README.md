# MSDA2025-7005-Group1-Assignment
Here is the assignment link for Group 1 of Course 7005 in the MSDA Programme 2025.

## 1.Background 

More than half of the world's people use the internet today (Boyd & Pennebaker, 2017) . Every post, comment, or message can reveal subtle clues about how a person thinks, feels, and interacts with others (Tausczik & Pennebaker, 2010). This makes language a powerful way to understand human behavior in the digital age.

One of the most well-known systems for describing personality is the Myers–Briggs Type Indicator, originally developed based on Carl Jung’s theory of psychological types; Myers 2016. The MBTI is a self-report questionnaire supposed to classify people into one of sixteen different personality types.

It emphasizes four key dimensions: Energy-preference for introversion or extroversion, Perceiving-preference for intuition or sensing, Judging-preference for thinking or feeling, and Orientation-preference for judging or perceiving. For each individual, the combination of those traits creates a four-letter code, such as INFP or ESTJ.

Regardless of debates concerning its scientific validity, MBTI is widely used in psychology, education, and business (Myers, 2016).


## 2.Research question: 
How well do supervised learning algorithms perform when predicting MBTI personality types from social media posts ?

## 3.Method 
### 3.1 Data cleaning
The dataset we used in this study is found on kaggle. It includes 8675 entries covering all 16 personality types with the correlated online posts. This dataset is a secondary dataset. The original dataset was collected via web scraping from social media platforms, resulting in inherent noise and low data quality issues.

<div align="center">
  <img src="https://raw.githubusercontent.com/Infernape819/MSDA2025-7005-Group1-Assignment/main/images/Dimension0.png" 
       alt="维度分析图"  
       width="800"/> 
  <p><em>Figure 1.</em>
</div>

<div align="center">
  <img src="https://raw.githubusercontent.com/Infernape819/MSDA2025-7005-Group1-Assignment/main/images/Dimension.png" 
       alt="维度分析图" 
       width="800"/> 
  <p><em>Figure 2. </em></p> 
</div>

Before formally proceeding with feature processing, we confirmed the basic characteristics of the data through a pie chart (Figure 1) and bar charts (Figure 2) depicting the distribution within each personality dimension 

As figure 1 and figure 2 show that ,there’s an obvious imbalance between MBTI personalities and within each dimensions. Therefore, before formally analyzing the dataset, we must prioritize using SMOTE from Python's imblearn package to oversample the minority class. This oversampling primarily involves repeatedly sampling the minority classes to ensure the model does not exhibit a preference bias for the majority class during classification. 

To scrub the noises in our text data, we first applied re and nltk packages on Python. Using the “re.sub” function, we efficiently removed punctuation, numbers, and other non-alphabetic characters from all posts. Next, we employed the “stopwords” and “lemmatzier” functions from the nltk package to eliminate all semantically insignificant stop words, such as “and,” and performed lemmatization. This process converted all word forms in the posts to their dictionary roots, facilitating subsequent analysis. 

After completing data cleaning, we applied TF-IDF processing to the cleaned dataset. This step primarily aimed to transform the original data into a sparse matrix with mostly zeros, which is a format more easily understood by machine learning models. By setting “max_feature” and “ngram_range,” we control the upper limit of feature count, preventing model overfitting and ensuring that paired phrases retain their original meaning when split. For example, “I love apple” can be decomposed into “I love” and “love apple” through parameter adjustment, rather than fragmented into “I,” “Love,” and “Apple.”

### 3.2 ML Models Used

After completing data cleaning and feature engineering, we first set up our baseline machine learning model: logistic regression. Here we applied two key tools from the scikit-learn package: “LogisticRegression” and “train_test_split, StratifiedKFold”. In the logistic regression model, we employed a baseline model and a model utilizing SMOTE oversampling. We conducted 10-fold cross-validation using the “StratifiedKFold” method. This involved training on 9 folds and validating on 1 fold iteratively, ensuring that the class proportions within each fold matched those of the original dataset. Given the inherent data imbalance, we evaluated model performance not only by accuracy but also by comprehensive metrics including precision, recall, and F1 score. 

To further enhance model performance, we additionally employed the “xgb” model tool from the xgboost package. In this tree-based model setting, we primarily adjusted “learning_rate” and “n_estimators” to control the complexity of each tree and their degree of influence on the final result. Just like in logistic regression, we employed SMOTE oversampling to balance the minority samples, along with K-fold cross-validation and multiple metrics to evaluate the model.

Finally, we attempted to use the deep learning model LSTM. For this step, we employed torchtext, transformers, and TensorFlow to process the data and build the model. First, we convert the cleaned text into integer sequences using TensorFlow's “tokenizer,” setting the vocabulary size to the top 10,000 most common words. Subsequently, we uniformly pad all texts to a length of 100 words using “pad_sequences”. These sequences are then processed into embeddings and continuously optimized during model training to capture semantic similarities between words. We then implemented a model capable of automatically extracting local patterns in text, identifying key phrase features, and dynamically assigning weights to words within sequences through an architecture combining convolutional neural networks, bidirectional long short-term memory networks, and attention mechanisms. 

To optimize the model, we employed TensorFlow's “EarlyStopping” function and “ReduceLROnPlateau” function. The former automatically halts training when model performance stops improving, while the latter adaptively reduces the learning rate during training stagnation. This approach helps the model escape digging local optima and discover superior solutions.


## 4. Result and conclusion
### 4.1 Overall Model Performance: Traditional Models Outperform Deep Learning

| Model           | Dimension | Accuracy | Minority Class | Minority Class | Key Insight                                  |
|-----------------|-----------|----------|----------------|----------------|----------------------------------------------|
| LR (SMOTE)      | I/E       | 0.85     | 0.68           | 0.69           | Highest F1 for I/E, high recall.             |
| XGBoost (SMOTE) | I/E       | 0.85     | 0.63           | 0.55           | Highest Accuracy, but lowest recall.         |
| Hybrid DL       | I/E       | 0.69     | 0.45           | 0.57           | Weakest performance in this dimension.       |
| LR (SMOTE)      | N/S       | 0.89     | 0.63           | 0.68           | Highest F1 and recall for N/S.               |
| XGBoost (SMOTE) | N/S       | 0.90     | 0.54           | 0.42           | Highest Accuracy, but lowest recall.         |
| Hybrid DL       | N/S       | 0.78     | 0.36           | 0.44           | Weakest performance overall.                 |
| LR (SMOTE)      | F/T       | 0.86     | 0.85           | 0.88           | Highest F1 and recall for F/T, most balanced.|
| XGBoost (SMOTE) | F/T       | 0.86     | 0.84           | 0.86           | Excellent performance, close to LR.          |
| Hybrid DL       | F/T       | 0.72     | 0.70           | 0.73           | Best-performing DL dimension.                |
| LR (SMOTE)      | J/P       | 0.81     | 0.74           | 0.72           | Highest F1 for J/P, high recall.             |
| XGBoost (SMOTE) | J/P       | 0.81     | 0.73           | 0.68           | Highest Accuracy, but lowest recall.         |
| Hybrid DL       | J/P       | 0.62     | 0.57           | 0.67           | Lowest Accuracy and F1.                      |
#### 4.1.1 Logistic Regression (LR): The Best Balance and Robustness

The LR (SMOTE) model achieved the highest Minority Class F1-Score and highest Recall across all four dimensions (I/E, N/S, F/T, J/P).

Conclusion: The combination of LR + Excellent Feature Engineering (TF-IDF) + SMOTE is the most stable and robust solution for this task, offering the highest recall (most effective at identifying minority types). It confirms that TF-IDF features are the most critical and effective representation for capturing MBTI language patterns in this dataset.

#### 4.1.2 XGBoost: The Highest Overall Accuracy

XGBoost achieved the highest overall accuracy in the I/E and N/S dimensions.This high accuracy comes at the cost of sacrificing Minority Class Recall (e.g., the lowest recall in I/E and N/S). XGBoost is conservative and prioritizes high precision (high Precision), meaning it accurately predicts a minority class when it does, but misses a large number of true minority samples.

#### 4.1.3 Hybrid DL Model: The Clear Underperformer

The Hybrid DL Model demonstrated the lowest accuracy and F1-Score across all four dimensions.

#### 4.1.4 Conclusion
Traditional deep learning architectures (LSTM/CNN with Attention) failed to extract more valuable, stable semantic information than the TF-IDF features used by the traditional models. Without incorporating large-scale pre-trained word vectors (like BERT), deep learning does not provide a clear advantage over optimized classical machine learning methods for this noisy, text-based personality prediction task.


### 4.2 Final Assessment of Dimension Prediction Difficulty
Based on the LR (SMOTE) F1-Scores (the most stable and balanced metric):

1. F/T (Feeling/Thinking) is the Easiest to Predict: This dimension consistently showed the best performance, indicating that the differences in emotional vs. logical language patterns are the most distinguishable.

2. N/S (Intuition/Sensing) is the Most Challenging: N/S remains the dimension with the lowest performance (F1 $\approx 0.63$) across all models. This confirms that the language expressing this dichotomy (abstract vs. concrete perception) is the most subtle and difficult to model using text features.

## 5. Limitations Analysis

1. Feature Representation Bottleneck:The high performance of LR and XGBoost relies heavily on TF-IDF. While this is the foundation of their success, it also defines their performance ceiling. To overcome the limitations in dimensions like N/S, introducing semantic features from modern, large-scale language models (e.g., fine-tuning BERT) is necessary.
   
3. Persistent Class Imbalance Trade-off:Although SMOTE improved minority class metrics, the models still force a trade-off between high Recall (favored by LR) and high Precision (favored by XGBoost). The inability of any model to achieve both simultaneously suggests that the inherent difficulty of the data distribution (class imbalance) remains a core challenge.
   
3. Inherent Discrepancy in Predictive Difficulty: The vast differences in F1-Scores across the four dimensions (from 0.85 to 0.63) are an intrinsic limitation of the task. It suggests that MBTI traits have differing levels of linguistic manifestation on social media. Traits like N/S are expressed in complex or subtle ways that are difficult to capture with current features.
  
4. Model Explainability: The LR model offers good interpretability (via coefficient weights). In contrast, the high-performing XGBoost (tree-based) and the Hybrid DL Model (black-box) have complex decision processes that are difficult to interpret. This is a drawback for a personality prediction task where behavior attribution analysis is often required.



# Reference 

Boyd, R. L., & Pennebaker, J. W. (2017). Language-based personality: A new approach to personality in a digital world. Current Opinion in Behavioral Sciences, 18, 63–68. https://doi.org/10.1016/j.cobeha.2017.07.017

Myers, S. (2016). Myers‐Briggs typology and Jungian individuation. Journal of Analytical Psychology, 61(3), 289–308. https://doi.org/10.1111/1468-5922.12233

Tausczik, Y. R., & Pennebaker, J. W. (2010). The Psychological Meaning of Words: LIWC and Computerized Text Analysis Methods. Journal of Language and Social Psychology, 29(1), 24–54. https://doi.org/10.1177/0261927X09351676
