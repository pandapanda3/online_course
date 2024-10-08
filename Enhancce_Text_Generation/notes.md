# Overview
Gemini is a family of generative AI models developed by Google DeepMind that is designed for multimodal use cases. The Gemini API gives you access to the Gemini Pro Vision and Gemini Pro models.

LangChain is a powerful open-source Python framework designed to make the development of applications leveraging large language models (LLMs) easier and more efficient.

Retrieval Augmented Generation (RAG) is a technique that allows language models to access and utilize external knowledge sources (commonly a vector database) when generating text, making their responses more accurate and informed.

## Documents
Generative AI introduction: 
https://cloud.google.com/vertex-ai/generative-ai/docs/learn/overview

Google Cloud Tech:
https://www.youtube.com/@googlecloudtech

GoogleCloudPlatform:
https://github.com/GoogleCloudPlatform/generative-ai

# Code 
## Configuration
```python
%%capture
!pip -q install langchain_experimental langchain_core
!pip -q install google-generativeai==0.3.1
!pip -q install google-ai-generativelanguage==0.4.0
!pip -q install langchain-google-genai
!pip -q install wikipedia
!pip -q install docarray
!pip -q install --upgrade protobuf google.protobuf
```

```python
import os
import google.generativeai as genai

from IPython.display import display
from IPython.display import Markdown
from google.protobuf.empty_pb2 import Empty

key_name = !gcloud services api-keys list --filter="gemini-api-key" --format="value(name)"
key_name = key_name[0]

api_key = !gcloud services api-keys get-key-string $key_name --location="us-central1" --format="value(keyString)"
api_key = api_key[0]

os.environ["GOOGLE_API_KEY"] = api_key

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
```

```python
# listing all the models
models = [m for m in genai.list_models()]
models
```

## Using Gemini directly with Python SDK
```python
# generate text
prompt = 'Who are you and what can you do?'

model = genai.GenerativeModel('gemini-pro')

response = model.generate_content(prompt)

Markdown(response.candidates[0].content.parts[0].text)
```

## Using Gemini with LangChain

### Basic LLM Chain

```python
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.7)


result = llm.invoke("What is a LLM?")

Markdown(result.content)
```

The text generated by the model is obtained by stream processing:
```python
for chunk in llm.stream("Write a poem about life. More than 300 words"):
    print(chunk.content)
    print("---")
```

### Basic Multi Chain

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser


model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
```

```python
prompt = ChatPromptTemplate.from_template(
    "tell me a short joke about {topic}"
)

output_parser = StrOutputParser()

chain = prompt | model | output_parser

chain.invoke({"topic": "machine learning"})
```
### A more complicated Chain - RAG
```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch

model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.7)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

from langchain.document_loaders import WikipediaLoader

# use Wikipedia loader to create some docs to use..
docs = WikipediaLoader(query="Machine Learning", load_max_docs=10).load()
docs += WikipediaLoader(query="Deep Learning", load_max_docs=10).load() 
docs += WikipediaLoader(query="Neural Networks", load_max_docs=10).load() 

# Take a look at a single document
docs[0]


```
```python
vectorstore = DocArrayInMemorySearch.from_documents(
    docs,
    embedding=embeddings # passing in the model to embed documents..
)

retriever = vectorstore.as_retriever()
```

```python
retriever.get_relevant_documents("what is machine learning?")
retriever.get_relevant_documents("what is gemini pro?")
```


```python
template = """Answer the question a a full sentence, based only on the following context:
{context}

Return you answer in three back ticks

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
```

```python
from langchain.schema.runnable import RunnableMap
retriever.get_relevant_documents("What is a graident boosted tree?")
chain = RunnableMap({
    "context": lambda x: retriever.get_relevant_documents(x["question"]),
    "question": lambda x: x["question"]
}) | prompt | model | output_parser
chain.invoke({"question": "What is machine learning?"})
chain.invoke({"question": "When was the transformer invented?"})
```