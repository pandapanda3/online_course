# Supervised Machine Learning and Logistic Regression

## Key Concepts
- **Goal**: Minimize the difference between actual labels (Y) and predictions (Ŷ) using a **cost function**.
- **Steps**:
  1. Run the prediction function.
  2. Minimize the cost by updating parameters iteratively.


## Sentiment Analysis Task
- **Example**: Predict if a tweet like *"I'm happy because I'm learning NLP"* is positive (1) or negative (0).
- **Classifier**: Logistic regression maps tweets to one of two classes.


## Steps to Build the Classifier
1. **Feature Extraction**: Process raw tweets to extract useful features.
2. **Model Training**: Train the logistic regression model to minimize cost.
3. **Prediction**: Classify tweets using the trained model.

# Text Representation as Vectors

## Key Concepts
- **Objective**: Encode text (e.g., tweets) as a numerical vector using a vocabulary.
- **Vocabulary (V)**: A list of unique words from all tweets.
  - Built by iterating through tweets and saving each unique word.
  - Example: "I am happy because I am learning" → Vocabulary: {I, am, happy, because, learning}.



## Feature Extraction
- For each word in the vocabulary:
  - **1** if the word appears in the tweet.
  - **0** if it does not appear.
- Result: A vector with **1s** for words present and **0s** for words absent.
  - Example: A tweet may have many **0s** and few **1s**, forming a **sparse representation**.

## Sparse Representation
- **Sparse Representation**:
  - High dimensionality with many zeros.
  - Features = Vocabulary size (**V**).
- **Challenges**:
  - Logistic regression must learn **V + 1 parameters**.
  - Large vocabularies lead to:
    - Long training times.
    - Slower predictions.


# Generating Word Counts for Logistic Regression

## Key Concepts
- **Objective**: Track word frequencies for positive and negative classes to extract features for logistic regression.
- **Frequency Dictionary**: A mapping from a word and its class (positive or negative) to the number of times it appears in that class.

## Steps to Generate Counts
1. **Define Vocabulary**:
   - Build a vocabulary of unique words from your corpus (e.g., 8 unique words in 4 tweets).
2. **Separate Tweets by Class**:
   - Divide tweets into **positive** and **negative** classes.
3. **Count Word Frequencies**:
   - For each word in the vocabulary, count occurrences in positive and negative tweets.
     - Example: 
       - "happy" appears **2 times** in positive tweets → Positive frequency = 2.
       - "am" appears **3 times** in negative tweets → Negative frequency = 3.

## Frequency Dictionary
- **Structure**: `{(word, class): frequency}`
  - Example:
    - `("happy", positive): 2`
    - `("am", negative): 3`
- **Purpose**: Used to extract features for logistic regression based on word occurrences in each class.

# Optimized Tweet Representation (Dimension 3)

## Key Concepts
- **Goal**: Reduce tweet representation from dimension **V** to dimension **3** for faster logistic regression.
- **Benefits**: 
  - Faster training and prediction.
  - Logistic regression only needs to learn 3 features instead of V.


## Features for Dimension 3 Representation
1. **Bias Unit**: A constant value of 1.
2. **Sum of Positive Frequencies**: Total frequency of words from the tweet in the positive class.
3. **Sum of Negative Frequencies**: Total frequency of words from the tweet in the negative class.


## Steps to Extract Features
1. **Prepare Frequency Dictionary**:
   - Map word-class pairs to their frequencies.
2. **Feature Extraction**:
   - **Second Feature**: Sum positive frequencies of words from the tweet.
   - **Third Feature**: Sum negative frequencies of words from the tweet.
   - Example:
     - For a tweet:
       - Sum of positive frequencies = 8.
       - Sum of negative frequencies = 11.
     - Resulting vector = `[1, 8, 11]`.

![Feature Extraction with Frequencies](image/Feature%20Extraction%20with%20Frequencies.png)

# Text Preprocessing: Stemming and Stop Words

## Key Steps
1. **Remove Stop Words and Punctuation**  
   - Eliminate common words (e.g., *and, are*) and punctuation.  
   - Remove handles (`@user`) and URLs.

2. **Stemming**  
   - Reduce words to their base form.  
     - Example: `tune`, `tuned`, `tuning` → `tun`

3. **Lowercasing**  
   - Convert all words to lowercase.  
     - Example: `GREAT`, `Great`, `great` → `great`

## Final Output
- Processed Tweet → `['tun', 'great', 'ai', 'model']`

# Building the Feature Matrix (X)

## Steps
1. **Preprocess Tweets**  
   - Remove stop words, URLs, handles, punctuation.  
   - Apply stemming and lowercase words.

2. **Extract Features**  
   - Use a frequency dictionary:  
     - **Bias**: 1  
     - **Positive Frequency Sum**  
     - **Negative Frequency Sum**

3. **Build X Matrix**  
   - Rows = Tweets (m tweets).  
   - Columns = `[1, Positive Sum, Negative Sum]`.

## Example
| Bias | Positive Sum | Negative Sum |
|------|--------------|--------------|
| 1    | 8            | 11           |
| 1    | 5            | 7            |

**Next**: Use `X` matrix in logistic regression.
![Code to Extract Feature](image/CodetoExtractFeature.png)