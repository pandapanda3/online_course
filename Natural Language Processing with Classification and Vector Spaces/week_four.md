# Word Vectors and Translation Using Matrices

## Overview
- **Word Vectors:** Capture semantic properties of words.
- **Goal:** Align word vectors from two different languages for basic translation.

## Translation Process
### Word Embedding Mapping
1. Generate embeddings for English and French words.
2. Transform English embeddings into the French embedding space.

### Matrix Transformation
- Use a transformation matrix \( R \) to map English word embeddings \( X \) to French embeddings \( Y \).
- Translate words by finding the most similar vectors in the French vector space.

## Steps to Implement
### Start with Random Matrix \( R \)
1. Compare \( X \times R \) with \( Y \).
2. Adjust \( R \) iteratively to minimize translation error.

### Subset Training
- Use a small set of English-French word pairs for training.
- Align vectors by ensuring corresponding rows in \( X \) and \( Y \) match.
- This model generalizes to words outside the training set.

## Optimization
### Loss Function
- Minimize \( ||X \times R - Y||_F^2 \) (squared Frobenius norm).
- Frobenius norm:
  - Sum of squared matrix elements, followed by square root.
- Squared Frobenius norm:
  - Avoids dealing with square roots, simplifies gradient calculation.

### Gradient Descent
1. Compute gradient of loss w.r.t. \( R \): \( \nabla_R = \frac{\partial \text{Loss}}{\partial R} \).
2. Update \( R \): \( R = R - \alpha \nabla_R \), where \( \alpha \) is the learning rate.
3. Iterate until loss falls below a threshold or after a fixed number of steps.

## Frobenius Norm Example
### Matrix Norm Calculation
- For matrix \( A = \begin{bmatrix} 2 & 2 \\ 2 & 2 \end{bmatrix} \):
  - Frobenius norm: \( \sqrt{2^2 + 2^2 + 2^2 + 2^2} = 4 \).
  - Squared norm: \( 2^2 + 2^2 + 2^2 + 2^2 = 16 \).

## Key Benefits
- Generalizes beyond the training subset.
- Efficient representation for translation tasks.

## Code Snippets
### Matrix Transformation
```python
import numpy as np
R = np.random.rand(2, 2)  # Define random matrix R
x = np.array([1, 2])      # Define vector x
transformed_vector = np.dot(x, R)  # Transform vector x using R
print(transformed_vector)
```


![Transforming word vectors](image/Transforming%20word%20vectors.png)

# Finding K-Nearest Neighbors and Hashing

## Overview
- **K-Nearest Neighbors (KNN):** Fundamental operation for finding similar word vectors, used in NLP tasks.
- **Transformation Space:** Transformed vectors may not exactly match any word vectors in the target space (e.g., French word vectors). KNN finds the closest match.

## Example: Finding Nearby Friends
1. **Sorting by Distance:**
   - Compare friend's locations to your location (e.g., San Francisco).
   - Rank friends by proximity.

2. **Efficient Searching:**
   - Instead of scanning all friends, filter by region (e.g., North America).
   - Search within the filtered subset for efficiency.

## Optimizing Search
- **Hash Tables:**
  - Organize data into buckets for faster lookup.
  - Efficient for grouping and searching data.

## Applications
- **Word Translation:**
  - Use KNN to find the closest word vector in the target language.
  - Example: Transform "hello" to French, finding similar vectors like "salut" or "bonjour."

- **Hashing:**
  - Introduced as a technique to optimize searches.
  - Faster than linear search, particularly for large datasets.

# Hash Tables and Hash Functions

## Overview
- **Hash Tables:** Data structures that organize data into buckets.
- **Hash Functions:** Functions that assign items to specific buckets based on a computed hash value.

## Example: Organizing a Cupboard
- Similar items are grouped into specific drawers (buckets):
  - Documents → One drawer
  - Keys → Another drawer
  - Books → A different drawer

## Concept of Buckets
- Data items are assigned to buckets based on a hash function.
- One bucket can hold multiple items, and each item is always assigned to the same bucket.

## Word Vectors and Hashing
- Word vectors can be assigned to buckets based on a hash value calculated using a hash function.
- Example:
  - Word vectors like 100, 14, 17, 10, and 97 are divided into buckets.
  - A modulo operation can be used as the hash function:
    - \( 14 \mod 10 = 4 \): Assigns 14 to bucket 4.
    - \( 17 \mod 10 = 7 \): Assigns 17 to bucket 7.

## Challenges with Basic Hashing
- Similar word vectors may not end up in the same bucket:
  - Example: \( 10, 14, 17 \) are assigned to different buckets.
- Ideal scenario: Group similar word vectors into the same bucket.

## Locality-Sensitive Hashing (LSH)
- **Definition:**
  - A specialized hashing method that ensures similar items are grouped together.
- **Key Features:**
  - Groups items based on their location in vector space.
  - Designed to handle proximity and similarity effectively.

## Key Concepts Learned
- **Hash Tables:** Efficient way to group and organize data.
- **Hash Functions:** Determine how items are distributed into buckets.
- **Limitations of Basic Hashing:** Does not guarantee grouping of similar items.
- **Locality-Sensitive Hashing:** A better approach for grouping similar items.

## Next Steps
- Explore **locality-sensitive hashing** for efficiently handling word vector similarity in bucket assignments.

# Locality Sensitive Hashing and Planes

## Overview
- **Locality Sensitive Hashing (LSH):** Technique for grouping data points into subsets based on location in vector space.
- **Planes in LSH:** Used to divide the space and classify vectors relative to these divisions.

## Key Concepts

### Planes and Vector Positioning
- **Planes:** Represent boundaries that divide the vector space.
  - A **plane** is defined by its **normal vector** (perpendicular to the plane).
  - Vectors are categorized as being on one side or the other based on their relationship with the plane.
- **Dot Product and Position:**
  - **Positive dot product:** Vector is on one side of the plane.
  - **Negative dot product:** Vector is on the opposite side.
  - **Zero dot product:** Vector lies on the plane.

### Example with Dot Products
- Given a normal vector \( P \):
  - **Vector 1:** Dot product \( P \cdot V1 = 3 \) → Positive, on one side of the plane.
  - **Vector 2:** Dot product \( P \cdot V2 = 0 \) → Lies on the plane.
  - **Vector 3:** Dot product \( P \cdot V3 = -3 \) → Negative, on the opposite side of the plane.

### Visualization of Dot Product
- **Projection:**
  - The dot product represents the projection of one vector onto another.
  - **Positive projection:** Parallel to the normal vector.
  - **Negative projection:** Opposite direction of the normal vector.
- **Significance:**
  - The sign of the dot product determines the vector's position relative to the plane.

### Practical Use in LSH
- **Planes help bucket vectors** based on their position:
  - Vectors on the same side of a plane are grouped together.
  - Multiple planes can further refine this grouping.

## Applications in Locality Sensitive Hashing
- Divide the vector space into regions using planes.
- Use the **dot product** and its sign to classify vectors into buckets.
- Combine multiple planes to approximate the position of data points more accurately.

## Key Takeaways
- The **sign of the dot product** tells you which side of the plane a vector lies.
- **Planes and normal vectors** are tools for dividing the space into meaningful subsets.
- LSH uses these concepts to efficiently group similar items.

![Locality sensitive hashing](image/Locality%20sensitive%20hashing.png)

# Combining Planes to Create a Hash Value

## Overview
- Locality Sensitive Hashing (LSH) divides vector space into regions using **multiple planes**.
- **Goal:** Assign each vector to a specific region by combining signals from planes into a single **hash value**.

## Key Concepts

### Using Multiple Planes
- Each plane determines whether a vector is on the positive or negative side based on the **sign of the dot product**:
  - Positive side → Intermediate hash value = 1.
  - Negative side → Intermediate hash value = 0.

### Combining Signals into a Single Hash Value
- **Steps:**
  1. Calculate the sign of the dot product for each plane.
  2. Assign an intermediate hash value (1 or 0) based on the sign.
  3. Combine intermediate hash values using the formula:
     \[
     Hash Value = 2^0 dot h_1 + 2^1 dot h_2 + 2^2 dot h_3 + \dots
     \]
     where \( h_1, h_2, h_3, \dots \) are the intermediate hash values.

### Example
- **Dot Products and Signs:**
  - Plane 1: Dot product = 3 → Positive → Intermediate hash = 1.
  - Plane 2: Dot product = 5 → Positive → Intermediate hash = 1.
  - Plane 3: Dot product = -2 → Negative → Intermediate hash = 0.
- **Combined Hash Value:**
  \[
  \text{Hash Value} = 2^0 \cdot 1 + 2^1 \cdot 1 + 2^2 \cdot 0 = 3
  \]

## Rules for Hashing
1. **Intermediate Hash Values:**
   - If dot product >= 0, assign h = 1 .
   - If dot product < 0 , assign h = 0.
2. **Combining Values:**
   - Use powers of 2 to combine intermediate hash values into a single hash value.

## Locality Sensitive Hashing in Practice
- **Purpose:**
  - Hash value identifies which bucket a vector belongs to in the divided vector space.
- **Benefits:**
  - Efficiently narrows down the search space for tasks like **k-nearest neighbors**.

## Key Takeaways
- Multiple planes divide vector space into smaller, manageable regions.
- **Hash values** are calculated by combining signals from all planes.
- LSH ensures vectors close in space are likely to have the same hash value, grouping similar items together.
- **Next Step:** Learn how LSH speeds up k-nearest neighbor computations.


![Multiple Planes](image/Multiple%20Planes.png)

# Approximate Nearest Neighbors with Locality Sensitive Hashing

## Overview
- Locality Sensitive Hashing (LSH) can speed up the search for **k-nearest neighbors**.
- Instead of brute force, LSH uses **multiple sets of random planes** to divide vector space into regions, making the search more efficient.

## Key Concepts

### Random Planes and Hash Tables
1. **Dividing Vector Space:**
   - A few planes can divide the space into regions, but there’s no way to determine the "best" planes.
   - Solution: Use **multiple sets of random planes**.
2. **Multiple Universes:**
   - Each set of random planes represents a different "universe."
   - Each universe creates independent hash tables, providing diverse perspectives for grouping similar vectors.

### Example of Random Planes in LSH
- **Process:**
  1. Consider a vector (e.g., magenta dot) representing a transformed word vector.
  2. Each set of random planes groups this vector into hash pockets:
     - **Universe 1:** Groups magenta with green vectors.
     - **Universe 2:** Groups magenta with blue vectors.
     - **Universe 3:** Groups magenta with orange vectors.
  3. Combining results from all universes gives a robust approximation of neighbors.

- **Outcome:** A subset of vectors that are possible candidates for the nearest neighbors.

### Approximate Nearest Neighbors
- **Efficiency vs. Precision:**
  - Searches only a subset of the vector space, not the entire space.
  - Results in **approximate** nearest neighbors, trading precision for speed.

## Advantages of Multiple Sets of Planes
- More robust identification of potential neighbors.
- Faster computation by focusing on smaller regions of vector space.

## Key Takeaways
- Locality Sensitive Hashing allows efficient computation of **approximate nearest neighbors**.
- Multiple sets of random planes divide vector space into independent hash tables.
- Results from all planes are combined to identify neighbors effectively.
- Sacrifices precision to gain substantial efficiency in large datasets.


![Approximate nearest neighbors](image/Approximate%20nearest%20neighbors.png)
# Document Search Using Fast k-Nearest Neighbors

## Overview
- Use **fast k-nearest neighbors** to search for text in a large collection of documents by comparing vectors.
- Represent both queries and documents as vectors and find the nearest neighbors.

## Representing Documents as Vectors
1. **From Words to Documents:**
   - Each document consists of multiple words (e.g., "I love learning").
   - Represent the document as a single vector by combining the word vectors.

2. **Steps to Create Document Vectors:**
   - Find the word vectors for each word in the document.
   - **Sum** the word vectors to form the document vector.
   - The resulting document vector has the same dimension as the individual word vectors.

## Applying k-Nearest Neighbors
- After creating document vectors for all documents in the collection:
  1. Represent the query as a vector using the same method.
  2. Compute the distances between the query vector and all document vectors.
  3. Identify the nearest neighbors (most similar documents).

## Implementation Summary
1. **Mini Dictionary for Word Embeddings:**
   - Store embeddings for words in a dictionary.
   - If a word is not in the dictionary, use a zero vector.
2. **Document Embedding:**
   - Initialize the document embedding as a vector of zeros.
   - For each word in the document:
     - Retrieve its embedding from the dictionary.
     - Add the embedding to the document vector.
   - Return the final document vector.

## Key Insights
- **Text Embedding:** Documents can be embedded into vector spaces for similarity search.
- **k-Nearest Neighbors:** Finds text with similar meaning based on vector closeness.
- **General Method:** This basic structure underpins many modern NLP techniques.
