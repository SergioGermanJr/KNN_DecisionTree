
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
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

# Modelo 1: Regressão Logística
print("Treinando Regressão Logística...")
param_grid_lr = {"C": [0.1, 1, 10]}
grid_search_lr = GridSearchCV(LogisticRegression(max_iter=200), param_grid_lr, cv=5)
grid_search_lr.fit(X_train, y_train)
best_lr = grid_search_lr.best_estimator_
print(f"Melhores parâmetros para Regressão Logística: {grid_search_lr.best_params_}")

# Modelo 2: Random Forest
print("Treinando Random Forest...")
param_grid_rf = {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]}
grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5)
grid_search_rf.fit(X_train, y_train)
best_rf = grid_search_rf.best_estimator_
print(f"Melhores parâmetros para Random Forest: {grid_search_rf.best_params_}")

# Modelo 3: Support Vector Machine (SVM)
print("Treinando SVM...")
param_grid_svm = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
grid_search_svm = GridSearchCV(SVC(random_state=42), param_grid_svm, cv=5)
grid_search_svm.fit(X_train, y_train)
best_svm = grid_search_svm.best_estimator_
print(f"Melhores parâmetros para SVM: {grid_search_svm.best_params_}")

# 2.4. Avaliação e Comparação de Modelos
print("\n2.4. Avaliação e Comparação de Modelos")

models = {
    "Logistic Regression": best_lr,
    "Random Forest": best_rf,
    "SVM": best_svm
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
plt.title("Comparação de Modelos de Machine Learning")
plt.ylabel("Score")
plt.ylim(0.8, 1.0)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("model_comparison.png")
print("\nGráfico de comparação de modelos salvo como model_comparison.png")

# Salvar resultados em um arquivo de texto
with open("results.txt", "w") as f:
    f.write("Métricas de Avaliação no Conjunto de Teste:\n")
    f.write(results_df.to_string())
print("Resultados salvos em results.txt")
