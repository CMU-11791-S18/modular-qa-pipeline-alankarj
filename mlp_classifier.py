from Classifier import Classifier
from sklearn.neural_network import MLPClassifier


# This is a subclass that extends the abstract class Classifier.
class MLP(Classifier):

    # The abstract method from the base class is implemented here
    # to return multinomial naive bayes classifier
    def buildClassifier(self, X_features, Y_train):
        clf = MLPClassifier().fit(X_features, Y_train)
        return clf
