from preprocessing import preprocess_heart_disease_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib  
X_train, X_test, y_train, y_test = preprocess_heart_disease_data()

clf = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [None, 4, 6, 8],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    estimator=clf,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_clf = grid_search.best_estimator_

test_score = best_clf.score(X_test, y_test)


model_filename = 'model.joblib'
joblib.dump(best_clf, model_filename)

loaded_model = joblib.load(model_filename)

loaded_model_score = loaded_model.score(X_test, y_test)
print("Test Set Accuracy (Loaded Model):", loaded_model_score)
