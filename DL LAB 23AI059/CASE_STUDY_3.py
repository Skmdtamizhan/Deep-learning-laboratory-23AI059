import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

data = {
    'email': [
        "Win a free lottery now",
        "Meeting tomorrow at 10am",
        "Limited offer, claim your prize",
        "Project deadline extended",
        "Congratulations! You won a gift",
        "Please review attached file",
        "Earn money quickly from home",
        "Team meeting agenda",
        "You have won a free ticket",
        "Submit your report by tonight"
    ],
    'label': [1,0,1,0,1,0,1,0,1,0]
}
df = pd.DataFrame(data)

X_train, X_test, y_train, y_test = train_test_split(
    df['email'], df['label'], test_size=0.3, random_state=42
)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

param_grid = {
    'C': [0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['liblinear']
}

grid = GridSearchCV(LogisticRegression(), param_grid, cv=2, scoring='accuracy')
grid.fit(X_train_vec, y_train)

print("Best Parameters:", grid.best_params_)
print("Best CV Score:", grid.best_score_)

y_pred = grid.predict(X_test_vec)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))
