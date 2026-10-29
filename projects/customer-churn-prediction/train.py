from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=1000, n_features=12, weights=[0.7, 0.3], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
model.fit(X_train, y_train)
print(classification_report(y_test, model.predict(X_test)))
