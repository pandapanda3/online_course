# RAG (Retrieval-Augmented Generation) Course Notes

## 1. Introduction & Background
- **Retrieve Relevant Documents**: Provide context to LLMs to improve answering and task performance.
- **Traditional Methods**: Simple retrieval based on semantic similarity or embeddings.
- **Improved Techniques**: Learn advanced methods to achieve significantly better results.

## 2. Key Challenges
- **Common Issue**: Retrieved documents may discuss similar topics but not contain the actual answer.
- **Solution**: **Query Expansion**  
  - Rewrite the query to pull in more directly relevant documents.  
  - Expand the query into multiple versions, guessing possible forms of the answer.

## 3. Advanced Techniques
1. **Query Optimization**  
   Use LLMs to improve the query itself.
2. **Re-Ranking Results**  
   Leverage a **Cross Encoder** to calculate relevance scores between sentence pairs.
3. **User Feedback**  
   Adapt query embeddings based on user input for improved precision.

## 4. Course Overview
- **Course Goals**:
  - Learn methods to enhance RAG systems.
  - Explore cutting-edge techniques and innovations.
- **Key Content**:
  - Identify pitfalls in basic retrieval approaches.
  - Improve results using LLMs, Cross Encoders, and user feedback.  
  - Discover emerging and experimental techniques just entering research.

