import sys
import json
import numpy as np
import pickle
from sklearn.externals import joblib

from Retrieval import Retrieval
from Featurizer import Featurizer
from CountFeaturizer import CountFeaturizer
from tfidf_featurizer import TFIDFFeaturizer

from Classifier import Classifier
from MultinomialNaiveBayes import MultinomialNaiveBayes
from svm_classifier import SupportVectorMachine
from mlp_classifier import MLP
from Evaluator import Evaluator
from sklearn.decomposition import TruncatedSVD


class Pipeline(object):
    def __init__(self, trainFilePath, valFilePath, retrievalInstance, featurizerInstance, classifierInstance):
        self.retrievalInstance = retrievalInstance
        self.featurizerInstance = featurizerInstance
        self.classifierInstance = classifierInstance
        trainfile = open(trainFilePath, 'r')
        self.trainData = json.load(trainfile)
        trainfile.close()
        valfile = open(valFilePath, 'r')
        self.valData = json.load(valfile)
        valfile.close()
        self.question_answering()

    def makeXY(self, dataQuestions):
        # print(json.dumps(dataQuestions[0], indent=2))
        # print(len(dataQuestions[2]["contexts"]["long_scores"]))
        X = []
        Y = []
        for i, question in enumerate(dataQuestions):
            # print("Question id: %d" % i)
            long_snippets = self.retrievalInstance.getLongSnippets(question)
            short_snippets = self.retrievalInstance.getShortSnippets(question)

            X.append(short_snippets)
            Y.append(question['answers'][0])

        return X, Y

    def question_answering(self):
        dataset_type = self.trainData['origin']
        candidate_answers = self.trainData['candidates']
        X_train, Y_train = self.makeXY(self.trainData['questions'])
        X_val, Y_val_true = self.makeXY(self.valData['questions'])

        # featurization
        X_features_train, X_features_val = self.featurizerInstance.getFeatureRepresentation(X_train, X_val)
        print(np.shape(X_features_train))
        print(np.shape(X_features_val))

        pca = TruncatedSVD(n_components=300)
        X_features_train = pca.fit_transform(X_features_train)
        X_features_val = pca.transform(X_features_val)

        self.clf = self.classifierInstance.buildClassifier(X_features_train, Y_train)

        # Prediction
        Y_val_pred = self.clf.predict(X_features_val)

        with open('naive-bayes_count.pkl', 'wb') as f:
            pickle.dump(Y_val_pred, f)

        self.evaluatorInstance = Evaluator()
        a = self.evaluatorInstance.getAccuracy(Y_val_true, Y_val_pred)
        p, r, f = self.evaluatorInstance.getPRF(Y_val_true, Y_val_pred)
        print("Accuracy: " + str(a))
        print("Precision: " + str(p))
        print("Recall: " + str(r))
        print("F-measure: " + str(f))


if __name__ == '__main__':
    trainFilePath = sys.argv[1]  # please give the path to your reformatted quasar-s json train file
    valFilePath = sys.argv[2]  # provide the path to val file
    retrievalInstance = Retrieval()
    featurizerInstance = CountFeaturizer()
    # featurizerInstance = TFIDFFeaturizer()
    # classifierInstance = MultinomialNaiveBayes()
    # classifierInstance = SupportVectorMachine()
    classifierInstance = MLP()
    trainInstance = Pipeline(trainFilePath, valFilePath, retrievalInstance, featurizerInstance, classifierInstance)
