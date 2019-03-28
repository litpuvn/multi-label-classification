import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection._split import StratifiedShuffleSplit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from sklearn.metrics import classification_report
from sklearn.svm.classes import SVC, LinearSVC

data_df = pd.read_csv("movies_genres_en.csv", delimiter='\t')


def grid_search(train_x, train_y, test_x, test_y, genres, parameters, pipeline):
    grid_search_tune = GridSearchCV(pipeline, parameters, cv=2, n_jobs=3, verbose=10)
    grid_search_tune.fit(train_x, train_y)

    print()
    print("Best parameters set:")
    print(grid_search_tune.best_estimator_.steps)
    print()

    # measuring performance on test set
    print("Applying best classifier on test data:")
    best_clf = grid_search_tune.best_estimator_
    predictions = best_clf.predict(test_x)

    print(classification_report(test_y, predictions, target_names=genres))



genres = list(data_df.drop(['title', 'plot'], axis=1).columns.values)
data_x = data_df[['plot']].as_matrix()
data_y = data_df.drop(['title', 'plot'], axis=1).as_matrix()
stratified_split = StratifiedShuffleSplit(n_splits=2, test_size=0.33)


x_train, x_test, y_train, y_test = train_test_split(data_x,data_y,test_size=0.33, random_state=42)

# transform matrix of plots into lists to pass to a TfidfVectorizer
train_x = [x[0].strip() for x in x_train.tolist()]
test_x = [x[0].strip() for x in x_test.tolist()]

stop_words = set(stopwords.words('english'))

## http://michelleful.github.io/code-blog/2015/06/20/pipelines/
## learn feature union to add more features (time, region)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=stop_words)),
    ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1)),
])
parameters = {
    'tfidf__max_df': (0.25, 0.5, 0.75),
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    "clf__estimator__C": [0.01, 0.1, 1],
    "clf__estimator__class_weight": ['balanced', None],
}
grid_search(train_x, y_train, test_x, y_test, genres, parameters, pipeline)