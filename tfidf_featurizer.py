from Featurizer import Featurizer
from sklearn.feature_extraction.text import TfidfVectorizer


# This is a subclass that extends the abstract class Featurizer.
class TFIDFFeaturizer(Featurizer):

    # The abstract method from the base class is implemented here to return count features
    def getFeatureRepresentation(self, X_train, X_val):
        count_vect = TfidfVectorizer(max_features=10000)
        X_train_counts = count_vect.fit_transform(X_train)
        X_val_counts = count_vect.transform(X_val)
        return X_train_counts, X_val_counts
