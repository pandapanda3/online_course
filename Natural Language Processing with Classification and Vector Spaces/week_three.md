# Notes on Constructing Vectors Using Co-occurrence Matrices

## Overview
- Learn how to construct vectors using **co-occurrence matrices** based on different tasks and designs.
- Encode words or documents as vectors.
- Explore **word-by-word** and **word-by-document** designs for vector space models.
- Understand relationships between words or documents using **similarity metrics**.

## Key Concepts

### Co-occurrence Matrix
- **Definition**: Counts how often two words appear together within a specified word distance (e.g., distance \( k \)).
- Example:
  - Sentence: "Data is simple."
  - Word "data" co-occurs with "simple" twice within distance \( k=2 \).
  - Vector for "data": `[2, 1, 1, 0]` (co-occurring with "simple," "raw," "like," "I").

### Word-by-Word Design
- **Output**: Vector with \( n \) entries (where \( n \) is vocabulary size).
- Represents co-occurrence of each word with other words in the corpus.

### Word-by-Document Design
- **Process**:
  - Count how often words appear in documents of specific categories.
- Example:
  - Word "data" appears:
    - Entertainment: 500 times.
    - Economy: 6,620 times.
    - Machine Learning: 9,320 times.
  - Word "film" appears:
    - Entertainment: 7,000 times.
    - Economy: 4,000 times.
    - Machine Learning: 1,000 times.
- Use the rows or columns to represent words or document categories as vectors.

### Vector Space
- **Representation**:
  - Vectors are created for words or document categories.
  - Example:
    - Economy and Machine Learning vectors are more similar than either is to Entertainment.
- **Similarity Measures**:
  - Compare vectors using:
    - **Cosine Similarity**: Angle between vectors.
    - **Euclidean Distance**: Distance between vectors.

## Applications
- Encode words or tweets into vectors.
- Solve tasks by selecting appropriate vector space designs.
- Determine relationships between documents or words using vector similarities.

![Word by Word and Word by Doc](image/Word%20by%20Word%20and%20Word%20by%20Doc.png)

# Euclidian Distance
![Euclidian Distance](image/Euclidian%20Distance.png)
# Cosine Similarity
- **Cosine Similarity**: A metric to determine the similarity between two vectors based on the cosine of the angle between them.
- **Advantage**: Not biased by the size differences between vector representations, unlike Euclidean distance.

## Problem with Euclidean Distance
- **Example**:
  - Comparing corpora represented by the words "disease" and "eggs."
  - Corpora:
    - **Food Corpus**: Fewer total words.
    - **Agriculture Corpus**: Similar total word count to History.
    - **History Corpus**: Larger total word count.
  - Observations:
    - **Distance \( d_1 \)** (Food ↔ Agriculture): Larger.
    - **Distance \( d_2 \)** (Agriculture ↔ History): Smaller.
    - Conclusion: Euclidean distance suggests Agriculture and History are more similar than Agriculture and Food.
- **Issue**: Euclidean distance is influenced by the **total word count** in corpora.

## Solution: Cosine Similarity
- **How it works**:
  - Measures the cosine of the angle between two vectors:
    - **Small Angle** → Cosine close to **1** (high similarity).
    - **Large Angle (approaching 90°)** → Cosine close to **0** (low similarity).
- **Example**:
  - **Angle \( \alpha \)** (Food ↔ Agriculture): Smaller, cosine close to 1.
  - **Angle \( \beta \)** (Agriculture ↔ History): Larger, cosine closer to 0.
  - Cosine similarity reflects actual similarity better than Euclidean distance.

## Key Takeaway
- **Cosine Similarity**:
  - Effective for comparing vectors of different sizes.
  - Focuses on the direction of vectors rather than their magnitude.
- **Use Case**: Ideal for comparing documents or corpora with varying word counts.


![Cosine Similarity](image/Cosine%20Similarity.png)
# Manipulating Vectors and Vector Arithmetic

## Overview
- **Vector Arithmetic**: Add or subtract vectors to infer relationships between words.
- **Key Use Case**: Predict unknown relationships using known relationships (e.g., countries and their capitals).

## Example: Predicting Capitals
1. **Known Data**:
   - Capital of the USA: Washington DC.
   - Capital of Russia: Unknown.
2. **Steps**:
   - **Find the relationship**: Compute the vector difference between Washington DC and USA:  
     \( \text{Washington DC} - \text{USA} \).
   - **Apply the relationship**: Add this difference vector to the vector for Russia:  
     \( \text{Russia} + (\text{Washington DC} - \text{USA}) \).
   - **Result**: The resulting vector points to the capital of Russia in the vector space.
   - **Match the closest vector**: Use **Euclidean distance** or **cosine similarity** to find the city closest to this vector. In this example, the closest city is **Moscow**.

## Importance of Vector Spaces
- **Relative Meaning**: Effective vector spaces encode relationships and capture semantic similarity.
- **Clustering**:
  - Words used in similar contexts will have similar vector representations.
  - Example: The word "doctor" will cluster with "doctors," "nurse," "cardiologist," "surgeon," etc.

## Applications
- **Inferring Unknown Relationships**: Use known relationships to predict unknown ones.
- **Identifying Patterns**: Take advantage of consistent vector encodings to identify related words or concepts.
- **Cosine Similarity**: Find closest vectors in meaning (e.g., similar professions or roles).

![Cosine Similarity](image/Cosine%20Similarity1.png)
![Cosine Similarity](image/Cosine%20Similarity2.png)
# Visualization and PCA

## Overview
- **Goal**: Reduce the dimensions of features while retaining as much information as possible.
- **Key Concepts**:
  - **Eigenvalues**: Represent the variance of data along new features.
  - **Eigenvectors**: Represent the directions of uncorrelated features.

## Steps to Perform PCA (Principal Component Analysis)

### 1. **Prepare Data**
- **Normalize**: Mean normalize your data to remove biases.
- **Covariance Matrix**: Calculate the covariance matrix of your data.

### 2. **Compute Eigenvalues and Eigenvectors**
- Perform **Singular Value Decomposition (SVD)** on the covariance matrix:
  - Results in three matrices:
    - **First Matrix**: Contains the Eigenvectors (stacked column-wise).
    - **Second Matrix**: Contains Eigenvalues (on the diagonal).
- **Ordering**:
  - Eigenvalues should be sorted in descending order.
  - Most libraries handle this automatically.

### 3. **Project Data to New Feature Space**
- Use the **Eigenvectors** and **Eigenvalues** to project data:
  - Select the first **n** columns of the Eigenvector matrix (where \( n \) is the desired dimensionality).
  - Perform dot product between:
    - Your original embeddings matrix.
    - The selected columns of the Eigenvector matrix.
- **Retain Variance**:
  - Ensure the projection retains the highest percentage of variance from the original data.

## Key Points
- **Uncorrelated Features**:
  - Eigenvectors provide directions for features that are uncorrelated.
- **Variance**:
  - Eigenvalues represent the variance of data along the new features.
- **Dimensionality Reduction**:
  - Common practice for visualization is to reduce data to 2 dimensions.


![Visualization and PCA](image/Visualization%20and%20PCA.png)
![PCA algorithm](image/PCA%20algorithm.png)
# The Rotation Matrix
![Counterclockwise Rotation](image/Counterclockwise%20Rotation.png)
![Clockwise Rotation](image/Clockwise%20Rotation.png)