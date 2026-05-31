import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_text
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


def visualize_classifier(model, X, y, ax=None, cmap='rainbow'):
    ax = ax or plt.gca()

    ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=cmap,
               vmin=y.min(), vmax=y.max(), zorder=3)
    ax.axis('tight')
    ax.axis('off')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    model.fit(X, y)
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
                         np.linspace(*ylim, num=200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=0.3,
                           levels=np.arange(n_classes + 1) - 0.5,
                           cmap=cmap, vmin=y.min(), vmax=y.max(),
                           zorder=1)

    ax.set(xlim=xlim, ylim=ylim)


DATA_PATH = Path(__file__).resolve().parent.parent / 'crop.csv'
df = pd.read_csv(DATA_PATH)
print(df.head())
print(df['Crop'].value_counts())

X = df.drop('Crop', axis=1)
y = df['Crop']

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)


tree = DecisionTreeClassifier().fit(X_train, y_train)

print(export_text(tree, feature_names=list(X.columns), max_depth=3))

fig = plt.figure(figsize=(25, 20))
plot_tree(tree, feature_names=list(X.columns), max_depth=3,
          class_names=list(le.classes_), filled=True)
plt.show()

y_pred = tree.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))


X2 = X[['Humidity', 'Rainfall']].values
visualize_classifier(DecisionTreeClassifier(), X2, y)
plt.show()


tree = DecisionTreeClassifier()
bag = BaggingClassifier(tree, n_estimators=100, max_samples=0.8,
                        random_state=1)
bag.fit(X_train, y_train)
print(accuracy_score(y_test, bag.predict(X_test)))

model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(X_train, y_train)
print(accuracy_score(y_test, model.predict(X_test)))

visualize_classifier(RandomForestClassifier(n_estimators=100, random_state=0),
                     X2, y)
plt.show()
