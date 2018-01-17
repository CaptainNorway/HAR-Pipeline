
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import numpy as np
from time import time


label_to_number_dict = {
    "none": 0,
    "walking": 1,
    "running": 2,
    "shuffling": 3,
    "stairs (ascending)": 4,
    "stairs (descending)": 5,
    "standing": 6,
    "sitting": 7,
    "lying": 8,
    "transition": 9,
    "transition_sitting_standing": 91,
    "tansition_standing_walking": 92,
    "transition_walking_standing": 93,
    "transition_standing_sitting": 94,
    "bending": 10,
    "picking": 11,
    "undefined": 12,
    "cycling (sit)": 13,
    "cycling (stand)": 14,
    "heel drop": 15,
    "vigorous activity": 16,
    "non-vigorous activity": 17,
    "Transport(sitting)": 18,
    "Commute(standing)": 19,
    "lying (prone)": 20,
    "lying (supine)": 21,
    "lying (left)": 22,
    "lying (right)": 23,
}


def getData(random):
    if (random):
        n_samples = 10000
        train_samples = int(0.7*n_samples)
        X, y = make_classification(n_samples=n_samples, n_features=138,
                              n_informative=40, n_redundant=98,
                               random_state=0, shuffle=False)

        train_X = np.array(X[:train_samples])
        train_y = np.array(y[:train_samples])
        test_X = np.array(X[train_samples:])
        test_y = np.array(y[train_samples:])

    else:
        train_X = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0],
                            [1, 1, 1], [1, 0, 1]])

        train_y = np.array([1, 0, 0, 0, 1, 1, 1, 0])
        train_y = np.array([0, 0, 0, 1, 1, 0, 1, 1])

        test_X = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [1, 1, 1]])
        test_y = np.array([0, 0, 0, 0, 1])

    return train_X, train_y, test_X, test_y


def getFeatureIndexes(feature_importances, features_top_percentage):
    number_of_features = int(features_top_percentage*len(feature_importances))
    print("Number of features wanted: ", number_of_features)
    feature_importances_sorted = np.sort(feature_importances)
    print("Feature importances sorted", feature_importances_sorted)
    feature_threshold = feature_importances_sorted[-number_of_features]
    print("Feature threshold: ", feature_threshold)

    feature_indexes = []
    for i in range(len(feature_importances)):
        if len(feature_indexes) < number_of_features:
            print("Feature importance: ", feature_importances[i], " feature threshold: ", feature_threshold)
            if feature_importances[i] >= feature_threshold:
                feature_indexes.append(i)
        else:
            break
    print("Number of features extracted: ", len(feature_indexes))
    return feature_indexes



def trainRFC(train_X, test_y):
    forest = RFC(n_estimators=100)
    forest.fit(train_X, train_y)
    print("Forest trained with ", forest.n_features_, " features")
    return forest

def testRFC(test_X, test_y, indexes):
    x=0

def getFeatures(data, indexes):
    print("Data set shape before reshape: ", data.shape)
    data = data[:, indexes]
    print("Data set shape afer reshape ", data.shape)
    return data






#Get data
random = 0
train_X, train_y, test_X, test_y = getData(random)
#Between 0 and 1
features_top_percentage = 0.4


#Create RFC for data with all features
forest_all_features = trainRFC(train_X, train_y)
#Create RFC for data with features where for each feature, feature importance
#is above or equal to threshold
feature_importances = forest_all_features.feature_importances_
print("Feature importances: ", feature_importances)
indexes = getFeatureIndexes(feature_importances, features_top_percentage)
train_X = getFeatures(train_X, indexes)
forest_best_features = trainRFC(train_X,train_y)

#Test procedure

#All features
a = time()
#a = time.time() * 1000
y_pred_all_features = forest_all_features.predict(test_X)
b = time()
#b=time.time() * 1000
print("TIME: Predict all features:", format(b - a, ".4f"), "s")
accuracy_all_features = accuracy_score(test_y, y_pred_all_features)
print("Accuracy for the RFC with all features = ", accuracy_all_features)
#Reduced feature set
test_X = getFeatures(test_X, indexes)
c = time()
#a = time.time() * 1000
y_pred_best_features = forest_best_features.predict(test_X)
d = time()
#b = time.time() * 1000

print("TIME: Predict best features:", format(d - c, ".4f"), "s")
accuracy_best_features = accuracy_score(test_y, y_pred_best_features)
print("Accuracy for the RFC with the best features = ", accuracy_best_features)


dict1 = {"1": [1,2], "2": [4,3]}
distribution_lists = sorted(dict1.items(), key=lambda dist:dist[1])
x,y = zip(*distribution_lists)
print(x)
print(y)
y = np.array(y)[:, [1]]
print(y)
print(type(y))
y.tolist()
print(type(y))