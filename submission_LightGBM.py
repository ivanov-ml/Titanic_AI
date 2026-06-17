import pandas as pd
import lightgbm as lgb

test_data = pd.read_csv('/Users/dmitrii/PycharmProjects/Titanic_AI_project/titanic/test.csv')
import joblib




passenger_ids = test_data['PassengerId']

test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1
test_data['IsAlone'] = (test_data['FamilySize'] == 1).astype(int)


test_data['IsChild'] = (test_data['Age'] < 12).astype(int)

test_data['WomanWithChild'] = ((test_data['Sex'] == 'female') & (test_data['Parch'] > 0)).astype(int)

test_data['FareBin'] = pd.qcut(test_data['Fare'], 4, labels=False)

test_data = test_data.drop(['Name', 'Cabin', 'Ticket', 'PassengerId'], axis=1)

test_data['Sex'] = test_data['Sex'].map({'male': 1, 'female': 0})
test_data['Embarked'] = test_data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
test_data['Age'].fillna(0, inplace=True)
test_data['Fare'].fillna(test_data['Fare'].median(), inplace=True)
test_data['Embarked'].fillna(0, inplace=True)
test_data['FareBin'].fillna(0, inplace=True)
test_data.columns = [f"col_{i}" for i in range(test_data.shape[1])]

data = pd.read_csv('/Users/dmitrii/PycharmProjects/Titanic_AI_project/data_clean_fe.csv')

if 'Unnamed: 0' in data.columns:
    data = data.drop(['Unnamed: 0'], axis=1)

X = data.drop(['Survived'], axis=1)
y = data['Survived']
X.columns = [f"col_{i}" for i in range(X.shape[1])]


model = lgb.LGBMClassifier(
    objective='binary',
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    reg_alpha=0.1,
    reg_lambda=0.2,
    min_child_samples=20,
    n_estimators=32,
    random_state=42,
    verbose=-1
)

model.fit(X, y)


test_predictions = model.predict(test_data)

submission = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Survived': test_predictions
})

submission.to_csv('submission.csv', index=False)
print("✅ submission.csv created!")

# После обучения pipeline
joblib.dump(model, 'models/titanic_pipeline.pkl')
print("Модель сохранена")
