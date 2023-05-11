import math
import pymorphy3
import re


class NaiveBayesClassifier:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.morph = pymorphy3.MorphAnalyzer()
        self.dictionary = {}
        self.labels = []
        self.labels_count = {}
        self.size = 0
        self.word_probability = {}

    def fit(self, X, y):
        """ Fit Naive Bayes classifier according to X, y. """
        self.labels = list(set(y))
        self.labels_count = dict.fromkeys(self.labels, 0)

        for label in y:
            self.labels_count[label] += 1

        for text, label in zip(X, y):
            words = [self.morph.parse(re.sub(r'[^\w\s]', '', word.lower()))[0].normal_form for word in text.split()]
            for word in set(words):
                if word not in self.dictionary:
                    self.dictionary[word] = dict.fromkeys(self.labels, 0)
                    self.word_probability[word] = dict.fromkeys(self.labels, 0)
                    self.size += 1
                self.dictionary[word][label] += words.count(word)

        for word in self.dictionary:
            for label in self.labels:
                nc = sum(self.dictionary[w][label] for w in self.dictionary)
                nic = self.dictionary[word][label]
                self.word_probability[word][label] = (nic + self.alpha) / (nc + self.size * self.alpha)

    def predict(self, X):
        """ Perform classification on an array of test vectors X. """
        results = []

        for text in X:
            probabilities = {
                label: math.log(self.labels_count[label] / sum(self.labels_count.values())) if self.labels_count[
                                                                                                   label] != 0 else -100000
                for label in self.labels}

            for word in set(text.split()):
                if word not in self.word_probability:
                    continue
                for label in self.labels:
                    probabilities[label] += math.log(self.word_probability[word][label])

            max_prob = max(probabilities.values())
            result = [label for label, prob in probabilities.items() if prob == max_prob]
            results.append(result[0])

        return results

    def score(self, X_test, y_test):
        """ Returns the mean accuracy on the given test data and labels. """
        correct_predictions = 0
        total_predictions = len(X_test)

        for i in range(total_predictions):
            prediction = self.predict([X_test[i]])[0]
            if prediction == y_test[i]:
                correct_predictions += 1

        return correct_predictions / total_predictions