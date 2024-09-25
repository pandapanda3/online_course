# Types of supervised learning
Before you use supervised learning

Requirements:

- No missing values
- Data in numeric format
- Data stored in pandas DataFrame or NumPy array
- Perform Exploratory Data Analysis (EDA) first

## Classification
Target variable consists of categories
## Regression
Target variable is continuous

# scikit-learn syntax
```python
from sklearn.module import Model
model = Model()
model.fit(X, y)
predictions = model.predict(X_new)
print(predictions)
```
# Classifying labels of unseen data
- Build a model
- Model learns from the labeled data we pass to it
- Pass unlabeled data to the model as input
- Model predicts the labels of the unseen data
- Labeled data = training data

```python
from sklearn.neighbors import KNeighborsClassifier
X = churn_df[["total_day_charge", "total_eve_charge"]].values
y = churn_df["churn"].values
print(X.shape, y.shape)
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X, y)

X_new = np.array([[56.8, 17.5], [24.4, 24.1], [50.1, 10.9]])
predictions = knn.predict(X_new)
print('Predictions: {}'.format(predictions))
```

# Measureing model performance
```python
from sklearn.model_selection import train_test_split
# stratify: ensures that the training and testing sets have the same proportion of each class label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))
```

# Model complexity and over/underfitting
- Larger k = less complex model = can cause underfitting
- Smaller k = more complex model = can lead to overfitting

```python
train_accuracies = {}
test_accuracies = {}
neighbors = np.arange(1, 26)
for neighbor in neighbors:    
    knn = KNeighborsClassifier(n_neighbors=neighbor)    
    knn.fit(X_train, y_train)    
    train_accuracies[neighbor] = knn.score(X_train, y_train)    
    test_accuracies[neighbor] = knn.score(X_test, y_test)

plt.figure(figsize=(8, 6))
plt.title("KNN: Varying Number of Neighbors")
plt.plot(neighbors, train_accuracies.values(), label="Training Accuracy")
plt.plot(neighbors, test_accuracies.values(), label="Testing Accuracy")
plt.legend()
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")
plt.show()
```

