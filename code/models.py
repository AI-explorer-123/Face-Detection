import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression


class WeakClassifier:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.inequality = None
        self.best_error = float('inf')

    def train(self, features, labels, weights, num_features, classifier_index, selected_features):
        labels = np.where(labels == 1, 1, -1)

        for feature_index in tqdm(range(num_features), desc=f'Training {classifier_index}th weak_classifier'):
            if feature_index in selected_features:
                continue
            feature_values = features[:, feature_index]
            sorted_indices = np.argsort(feature_values)
            sorted_features = feature_values[sorted_indices]
            sorted_labels = labels[sorted_indices]
            sorted_weights = weights[sorted_indices]

            lt_errors = self._compute_error(
                sorted_weights, sorted_labels, 'lt')
            self._update_para(lt_errors, 'lt', sorted_features, feature_index)
            gt_errors = self._compute_error(
                sorted_weights, sorted_labels, 'gt')
            self._update_para(gt_errors, 'gt', sorted_features, feature_index)

        print(
            f"The {classifier_index}th classsifier choose {self.feature_index}th feature, inequality={self.inequality}")

        best_predictions = self.predict(features)
        return self.best_error, best_predictions, self.feature_index

    def predict(self, features):
        feature_values = features[:, self.feature_index]
        if self.inequality == 'lt':
            return np.where(feature_values < self.threshold, 1, 0)
        else:
            return np.where(feature_values > self.threshold, 1, 0)

    def _compute_error(self, weights, labels, inequality):
        if inequality == 'lt':
            errors_left = np.cumsum(weights * (labels == -1))
            errors_left = np.concatenate((np.array([0]), errors_left[:-1]))
            errors_right = np.cumsum(
                weights[::-1] * (labels[::-1] == 1))[::-1]
            errors_right = np.concatenate((errors_right[1:], np.array([0])))
            rest_errors = errors_left + errors_right
        else:
            errors_left = np.cumsum(weights * (labels == 1))
            errors_left = np.concatenate((np.array([0]), errors_left[:-1]))
            errors_right = np.cumsum(
                weights[::-1] * (labels[::-1] == -1))[::-1]
            errors_right = np.concatenate((errors_right[1:], np.array([0])))

            rest_errors = errors_left + errors_right
        return rest_errors + weights * (labels != -1)

    def _update_para(self, errors, inequality, features, feature_index):
        if np.min(errors) < self.best_error:
            min_error_index = np.argmin(errors)
            self.best_error = errors[min_error_index]
            self.feature_index = feature_index
            self.threshold = features[min_error_index]
            self.inequality = inequality
            # print(feature_index)


class AdaBoost:
    def __init__(self, weak_classifier=WeakClassifier, num_classifiers=32):
        self.num_classifiers = num_classifiers
        self.weak_classifier = weak_classifier
        self.classifiers = []
        self.alphas = []
        self.selected_features = []

    def train(self, features, labels, train_mask, test_mask):
        train_features = features[train_mask]
        test_features = features[test_mask]
        train_labels = labels[train_mask]
        test_labels = labels[test_mask]

        num_samples, num_features = train_features.shape
        positive_count = np.sum(train_labels)
        negative_count = np.sum(np.where(train_labels == 0, 1, 0))
        weights = np.where(train_labels == 1,
                           1/(2*positive_count),
                           1/(2*negative_count))

        print('Training Adaboost...')
        for i in range(self.num_classifiers):
            weak_classifier = self.weak_classifier()
            error, predictions, selected_feature = weak_classifier.train(
                train_features, train_labels, weights, num_features, i+1, self.selected_features)
            self.selected_features.append(selected_feature)
            alpha = np.log((1 - error) / error)
            weights = self._update_weights(
                weights, train_labels, predictions, alpha)
            self.alphas.append(alpha)
            self.classifiers.append(weak_classifier)

            self.evaluate(train_features, train_labels, 'train')
            self.evaluate(test_features, test_labels, 'test')

    def evaluate(self, features, labels, mode):
        y_pred = self.predict(features)
        y_true = labels
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        print(f"{mode}_set: f1: {f1:.4f}, auc: {auc:.4f}, acc: {acc:.4f}")

    def predict(self, features):
        num_samples = features.shape[0]
        final_predictions = np.zeros(num_samples)

        for classifier, alpha in zip(self.classifiers, self.alphas):
            predictions = classifier.predict(features)
            final_predictions += predictions * alpha
        return final_predictions >= 0.5*np.sum(np.array(self.alphas))

    def predict_proba(self, features):
        num_samples = features.shape[0]
        final_predictions = np.zeros(num_samples)

        for classifier, alpha in zip(self.classifiers, self.alphas):
            predictions = classifier.predict(features)
            final_predictions += predictions * alpha
        return (final_predictions/np.sum(np.array(self.alphas)))

    def _update_weights(self, weights, labels, predictions, alpha):
        weights *= np.exp(-alpha * np.where(labels == predictions, 1, 0))
        weights /= np.sum(weights)
        return weights


class Logistic:
    def __init__(self):
        self.model = LogisticRegression()
    
    def train(self, features, labels, train_mask, test_mask):
        train_features = features[train_mask]
        test_features = features[test_mask]
        train_labels = labels[train_mask]
        test_labels = labels[test_mask]

        self.model.fit(train_features, train_labels)
        self.evaluate(train_features, train_labels, 'train')
        self.evaluate(test_features, test_labels, 'test')

    def predict(self, features):
        return self.model.predict(features)

    def predict_proba(self, features):
        return self.model.predict_proba(features)[0][1]

    def evaluate(self, features, labels, mode):
        y_pred = self.predict(features)
        y_true = labels
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        print(f"{mode}_set: f1: {f1:.4f}, auc: {auc:.4f}, acc: {acc:.4f}")
