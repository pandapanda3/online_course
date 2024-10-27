# What is MLflow?

**MLflow** is an open-source platform designed to manage the entire machine learning (ML) lifecycle, from experimentation to deployment. It offers tools for tracking experiments, packaging models, and managing deployments, helping data scientists and ML engineers streamline the development of machine learning applications. 

MLflow consists of the following key components:

### 1. **MLflow Tracking**
   - **Purpose**: It allows you to log and query experiments, including parameters, metrics, and artifacts (such as models or datasets).
   - **Functionality**: 
     - Tracks and logs experiments with different machine learning models.
     - Stores metrics such as accuracy, loss, or custom metrics defined by the user.
     - Supports logging data to various backends such as local filesystems or remote databases.

### 2. **MLflow Projects**
   - **Purpose**: It standardizes how to package machine learning code in a reproducible manner.
   - **Functionality**:
     - Each ML project contains code and dependencies, making it easy to share and reproduce experiments.
     - Projects can be defined using a simple YAML file (`MLproject`), specifying dependencies and the command to run the experiment.

### 3. **MLflow Models**
   - **Purpose**: It provides a standardized way to package and serve machine learning models.
   - **Functionality**:
     - Supports multiple ML frameworks (e.g., TensorFlow, PyTorch, Scikit-learn).
     - Models can be served using APIs or deployed to various platforms such as AWS SageMaker, Azure ML, etc.

### 4. **MLflow Model Registry**
   - **Purpose**: It helps manage the lifecycle of machine learning models, including versioning, deployment, and stage transitions.
   - **Functionality**:
     - Tracks model versions, allowing easy rollback and comparison between different models.
     - Maintains a registry where models can be marked as "staging," "production," or "archived."

### Key Benefits of MLflow:
- **Flexibility**: Works with any ML library, algorithm, or programming language.
- **Reproducibility**: Ensures that experiments can be easily reproduced by tracking the code, parameters, and environment.
- **Scalability**: Suitable for both small-scale projects and enterprise-level deployments.
- **Integrations**: Supports integrations with popular ML frameworks (e.g., Scikit-learn, TensorFlow, PyTorch) and platforms (e.g., AWS, Azure).

### Example Use Cases:
1. **Tracking Experiments**: Log parameters, metrics, and artifacts during training to track performance across different runs.
2. **Serving Models**: Deploy a trained model using MLflow’s API to create a REST endpoint for prediction.
3. **Version Control**: Manage different versions of models and easily transition between staging and production environments.

### Summary
MLflow is a versatile tool that simplifies the development, tracking, and deployment of machine learning models. By providing a unified interface to manage experiments, code, and models, it helps teams work more efficiently and scale their ML operations.

For more information, visit the [official MLflow documentation](https://mlflow.org/).

# Install MLflow
* conda create -n mlflow-env python=3.8
* conda activate mlflow-env
* conda install -c conda-forge mlflow
* mlflow --version
* conda env list # check the env list

# Run the server
* mlflow ui
* open the link: http://127.0.0.1:5000
## Run local service
Just run the service here, and check the information in http://127.0.0.1:5000
## Run remote service
* mlflow run git@github.com:databricks/mlflow-example.git -P alpha=5
* check the information in http://127.0.0.1:5000

# Three main components

- **MLflow Tracking** - Logs key metrics, parameters, models, and other artifacts when running ML code to monitor experiments.
- **MLflow Projects** - Configurable standard format for organizing ML code to ensure consistency and reproducibility.
- **MLflow Models** - Packages ML model files with their dependencies so they can be deployed on diverse platforms.
