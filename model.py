# Import sklearn models

from nltk import SnowballStemmer
from sklearn import linear_model
from sklearn import svm, neighbors, tree, neural_network, feature_extraction, pipeline, metrics, model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.svm import SVC


class StemmedCountVectorizer(feature_extraction.text.CountVectorizer):

    def build_analyzer(self):
        stemmer = SnowballStemmer("english")
        analyzer = feature_extraction.text.CountVectorizer(analyzer='word',
                                                           max_features=1200,
                                                           ngram_range=(1, 2),
                                                           tokenizer=None,
                                                           preprocessor=None,
                                                           stop_words=None
                                                           ).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])


class Model:

    def __init__(self, X_train, y_train, labels):
        self._X_train = X_train
        self._y_train = y_train

        self._labels = labels

        self._configure_classifier()

    def _configure_classifier(self):
        self._models = {
            # using Radial Basis Function kernel
            'SVC_RBF': svm.SVC(kernel="rbf", gamma=1, class_weight='balanced', random_state=42),
            'SVC_SIGMOID': svm.SVC(kernel="sigmoid", gamma=1, C=1, degree=2, class_weight='balanced', random_state=47),
            'SVC_LINEAR': SVC(kernel="linear", gamma=3, C=1, class_weight='balanced'),
            'SVM': svm.LinearSVC(C=.5, class_weight='balanced'),
            'LOGISTIC_REGRESSION': linear_model.LogisticRegression(solver='lbfgs', class_weight='balanced', random_state=47),
            'NEAREST_CENTROID': NearestCentroid(),
            'KNN': neighbors.KNeighborsClassifier(n_neighbors=50, algorithm='auto', p=2),
            'DECISION_TREE': tree.DecisionTreeClassifier(),
            'RANDOM_FOREST': RandomForestClassifier(),
            'NEURAL_NETWORK': neural_network.MLPClassifier(solver='sgd', alpha=1e-2, hidden_layer_sizes=(5, 2), random_state=42),
        }

    def prediction_with_model_name(self, model, test_data):
        print("Building model with classifier %s" % model)

        vectorizer = StemmedCountVectorizer()

        classifier = pipeline.Pipeline([
            ('VECTORIZATION', vectorizer),
            ('TFIDF_TRANSFORMATION', feature_extraction.text.TfidfTransformer()),
            ('CLASSIFICATION', self._models[model])
        ])

        classifier = classifier.fit(self._X_train, self._y_train)

        return classifier.predict(test_data)

    def compute_metrics(self, predictions):
        class_metrics = metrics.precision_recall_fscore_support(self._y_train, predictions, labels=self._labels)

        precision = metrics.precision_score(self._y_train, predictions, labels=self._labels, average='macro')
        recall = metrics.recall_score(self._y_train, predictions, labels=self._labels, average='macro')
        f_score = metrics.f1_score(self._y_train, predictions, labels=self._labels, average='macro')
        accuracy = metrics.accuracy_score(self._y_train, predictions)

        return accuracy, class_metrics, f_score, precision, recall

    def model_accuracy(self, models):
        vectorizer = StemmedCountVectorizer()

        resp = {}

        for model in models:
            print("Running model for %s" % model)

            classifier = pipeline.Pipeline([
                ('VECTORIZATION', vectorizer),
                ('TFIDF_TRANSFORMATION', feature_extraction.text.TfidfTransformer()),
                ('CLASSIFICATION', self._models[model])
            ])

            predictions = model_selection.cross_val_predict(classifier, self._X_train, self._y_train, n_jobs=-1, cv=10)

            accuracy, class_metrics, f_score, precision, recall = self.compute_metrics(predictions)

            print(
              "---- %s ----\n Class metrics: %s\nPrecision: %s\nRecall: %s\nF_Score: %s\nAccuracy: %s\n-----------\n" % (
                  model, class_metrics, precision, recall, f_score, accuracy))

            resp[model] = (accuracy, class_metrics, f_score, precision, recall)

        return resp
