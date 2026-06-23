import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import lightgbm as lgb
import mlflow
import mlflow.sklearn

# Устанавливаем tracking URI на запущенный сервер MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

data = pd.read_csv('/Users/dmitrii/PycharmProjects/Titanic_AI_project/data_clean_fe.csv')

if 'Unnamed: 0' in data.columns:
    data = data.drop(['Unnamed: 0'], axis=1)

X = data.drop(['Survived'], axis=1)
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Параметры модели
model_params = {
    'objective': 'binary',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'reg_alpha': 0.1,
    'reg_lambda': 0.2,
    'min_child_samples': 20,
    'n_estimators': 32,
    'random_state': 42,
    'verbose': -1
}

# Запускаем MLflow эксперимент
with mlflow.start_run(run_name="titanic_lightgbm"):
    # Логируем параметры модели
    mlflow.log_params(model_params)

    # Создаём и обучаем модель
    model = lgb.LGBMClassifier(**model_params)
    model.fit(X_train, y_train)

    # Предсказания
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # Метрики
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Логируем метрики
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("roc_auc", auc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    # Вывод в консоль
    print(f"Accuracy: {accuracy:.3f}")
    print(f"ROC AUC: {auc:.3f}")
    print(f"Precision-Recall: {precision:.3f}-{recall:.3f}")

    # Кросс-валидация
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
    cv_mean = scores.mean()
    cv_std = scores.std()

    # Логируем кросс-валидацию
    mlflow.log_metric("cv_accuracy_mean", cv_mean)
    mlflow.log_metric("cv_accuracy_std", cv_std)

    print(f"Cross-validation accuracy: {cv_mean:.4f} (+/- {cv_std:.4f})")

    # Сохраняем модель в MLflow
    mlflow.sklearn.log_model(model, "titanic_lightgbm_model")

print("✅ Эксперимент залогирован в MLflow")