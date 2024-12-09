# Introduction to LangChain and Key Components

## LangChain Overview
LangChain is a Python framework for AI development, enabling real-time data processing and integration with Large Language Models (LLMs). It streamlines communication, vector embedding, and LLM interactions.

## Core Components
### Prompt Templates
- Structure input prompts to optimize LLM outputs for tasks like Q&A and summaries.

### LLMs
- Integrates models like GPT-3 and BLOOM from OpenAI or Hugging Face for text generation.

### Agents
- Combine LLMs with decision-making to automate tasks like searches and data processing.

## Summarization Methods
### Stuffing
- Provides all data to the LLM in a single call. 
- Fast and maintains context but limited by model's context window.

### MapReduce
- Summarizes chunks of text in stages, overcoming context limits.
- Requires multiple calls and may lose context between chunks.

## Prompt Templates in LangChain
- Use `PromptTemplate()` to create structured prompts with placeholders and few-shot examples for better LLM responses.

## Streamlit
- A Python framework for building and sharing machine learning web apps, enabling rapid development and deployment.
