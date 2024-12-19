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