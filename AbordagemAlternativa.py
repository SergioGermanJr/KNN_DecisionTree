import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# 2.2. Obtenção e Preparação dos Dados
print("2.2. Obtenção e Preparação dos Dados")
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# Dividir os dados em conjuntos de treino, validação e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train) # 0.25 of 0.8 is 0.2

print(f"Tamanho do conjunto de treino: {len(X_train)} amostras")
print(f"Tamanho do conjunto de validação: {len(X_val)} amostras")
print(f"Tamanho do conjunto de teste: {len(X_test)} amostras")

# 2.3. Seleção e Treinamento de Modelos
print("\n2.3. Seleção e Treinamento de Modelos")

# Modelo 1: K-Nearest Neighbors
print("Treinando K-Nearest Neighbors...")
param_grid_knn = {"n_neighbors": [3, 5, 7, 9]}
grid_search_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5)
grid_search_knn.fit(X_train, y_train)
best_knn = grid_search_knn.best_estimator_
print(f"Melhores parâmetros para K-Nearest Neighbors: {grid_search_knn.best_params_}")

# Modelo 2: Decision Tree
print("Treinando Decision Tree...")
param_grid_dt = {"max_depth": [None, 5, 10, 15], "min_samples_leaf": [1, 2, 4]}
grid_search_dt = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_dt, cv=5)
grid_search_dt.fit(X_train, y_train)
best_dt = grid_search_dt.best_estimator_
print(f"Melhores parâmetros para Decision Tree: {grid_search_dt.best_params_}")

# 2.4. Avaliação e Comparação de Modelos
print("\n2.4. Avaliação e Comparação de Modelos")

models = {
    "K-Nearest Neighbors": best_knn,
    "Decision Tree": best_dt
}

results = {}

for name, model in models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    results[name] = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1-Score": f1}

results_df = pd.DataFrame(results).T
print("\nMétricas de Avaliação no Conjunto de Teste:")
print(results_df)

# Visualização dos resultados
results_df.plot(kind="bar", figsize=(10, 6))
plt.title("Comparação de Modelos de Machine Learning (Alternativa)")
plt.ylabel("Score")
plt.ylim(0.8, 1.0)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("model_comparison_alternative.png")
print("\nGráfico de comparação de modelos salvo como model_comparison_alternative.png")

# Salvar resultados em um arquivo de texto
with open("results_alternative.txt", "w") as f:
    f.write("Métricas de Avaliação no Conjunto de Teste (Alternativa):\n")
    f.write(results_df.to_string())
print("Resultados salvos em results_alternative.txt")
