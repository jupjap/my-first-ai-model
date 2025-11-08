# model.py
# Simple AI / ML model for IT Career Switch submission

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. Load data
iris = load_iris()
X = iris.data     # features
y = iris.target   # labels

# 2. Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Create the model
model = DecisionTreeClassifier(random_state=42)

# 4. Train the model
model.fit(X_train, y_train)

# 5. Make predictions
y_pred = model.predict(X_test)

# 6. Evaluate
acc = accuracy_score(y_test, y_pred)

print("Model trained successfully!")
print(f"Accuracy: {acc:.2f}")

# 7. Make a single prediction (example)
sample = X_test[0].reshape(1, -1)
pred_class = model.predict(sample)[0]
print(f"Example prediction class: {pred_class} ({iris.target_names[pred_class]})")
