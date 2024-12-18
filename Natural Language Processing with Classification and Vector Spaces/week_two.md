# Bayes Rule Overview and Probability Basics
There are many applications of naive Bayes including: 

* Author identification
* Spam filtering 
* Information retrieval 
* Word disambiguation 


## Key Points:
1. **Bayes Rule Introduction**:
   - Widely used in NLP for tasks like sentiment analysis and auto-correction.
   - Derived from the definition of conditional probabilities.

2. **Main Topics**:
   - Basics of probability and conditional probability.
   - Theory and practical applications of Bayes Rule.

3. **Applications**:
   - Sentiment analysis in NLP.
   - Auto-correction implementation in later courses.

4. **Highlights**:
   - Understand fundamental probability principles.
   - Learn to derive and apply Bayes Rule.
![Bayes Rule](image/Bayes.png)
# Naive Bayes Introduction

![Naive Bayes Compute](image/Naive%20Bayes%20Compute.png)
![Naive Bayes Rule](image/Naive%20Bayes.png)
# Laplacian Smoothing
By adding V (the number of unique words in the vocabulary) to the denominator and adding 1 to the numerator, the result will never be zero, as illustrated in the diagram.
![Laplacian Smoothing](image/Laplacian%20Smoothing.png)
![smoothing compute](image/smoothing%20compute1.png)
![smoothing compute](image/smoothing%20compute2.png)

# Log Likelihood
![Log Likelihood](image/Log%20Likelihood1.png)
![Log Likelihood](image/Log%20Likelihood2.png)
![Log Likelihood](image/Log%20Likelihood3.png)
![Log Likelihood](image/Log%20Likelihood4.png)

# Training naïve Bayes
![Training naïve Bayes](image/Training%20naïve%20Bayes.png)

# Testing naïve Bayes
![Testing naïve Bayes](image/Testing%20naïve%20Bayes.png)
![Testing naïve Bayes](image/Testing%20naïve%20Bayes1.png)
![Testing naïve Bayes](image/Testing%20naïve%20Bayes2.png)

# Assumptions of Naïve Bayes Method

## Key Assumptions:
1. **Independence of Words**:
   - Naïve Bayes assumes words in a sentence are **independent** of one another.
   - **Issue**: Words often occur together (e.g., *sunny* and *hot*) and are contextually related, which violates this assumption.
   - Consequence: Conditional probabilities of individual words may be **underestimated** or **overestimated**.

2. **Example of Independence Problem**:
   - Sentence completion task: "It’s always cold and snowy in ___."
     - Naïve Bayes might assign equal probabilities to *spring*, *summer*, *fall*, and *winter*.
     - Contextually, *winter* is clearly the most likely word, but Naïve Bayes does not capture this.


## Validation and Training Data:
1. **Balanced vs. Real Data**:
   - Naïve Bayes relies on the distribution of the training data.
   - Real-world datasets (e.g., tweet streams) are often **imbalanced**:
     - Positive tweets occur more frequently than negative tweets.
     - Negative tweets may include banned or muted content (e.g., offensive vocabulary).

2. **Impact of Imbalanced Data**:
   - Artificially balanced datasets (like those used in assignments) may cause the model to behave **optimistically** or **pessimistically** when applied to real-world data.


## Recap:
- **Independence Assumption**: Difficult to guarantee, as words are often contextually related.
- **Training Data**: Balanced data is necessary for accurate results.
- Despite these assumptions, Naïve Bayes can perform well in certain scenarios.

# Analyzing Errors in Naïve Bayes for NLP

## Common Sources of Errors:
1. **Semantic Meaning Lost in Preprocessing**:
   - Example: Removing punctuation can alter sentiment.
     - Original: *"My beloved grandmother :("* → Sad sentiment.
     - Processed: *"beloved grandmother"* → Positive sentiment (incorrect).

2. **Word Order Matters**:
   - Naïve Bayes ignores word order, leading to incorrect predictions.
     - Example:
       - *"I’m happy because I did not go"* → Positive.
       - *"I’m not happy because I did not go"* → Negative.  
     - Removing words like "not" can misrepresent the sentiment.

3. **Neutral Words Removal**:
   - Removing neutral words can distort meaning.
     - Example:
       - Original: *"This is not good because your attitude is not even close to being nice."*
       - Processed: *"Good, attitude, close, nice"* → Positive sentiment (incorrect).

4. **Language Quirks**:
   - Naïve Bayes struggles with:
     - **Sarcasm**, **irony**, and **euphemisms** (adversarial attacks).
   - Example:
     - Tweet: *"This is a ridiculously powerful movie. The plot was gripping, and I cried right through until the ending."*
     - Preprocessing leaves mostly negative words → Incorrect negative prediction.


## Key Takeaways:
1. **Check Processed Text**:
   - Always review the text after preprocessing to ensure critical information (e.g., punctuation, negations) is not lost.

2. **Word Order Matters**:
   - Naïve Bayes assumes word independence, ignoring order, which can lead to misclassifications.

3. **Adversarial Language**:
   - Handle sarcasm, irony, and euphemisms carefully, as machines struggle to interpret these.


## Conclusion:
- Naïve Bayes makes **independence assumptions**, leading to errors in some cases.
- Despite its limitations, it remains a **powerful baseline model** due to its simplicity and reliance on word frequency counts.
