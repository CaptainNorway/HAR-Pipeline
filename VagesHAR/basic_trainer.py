from collections import Counter

import warnings
import os
import numpy as np
import pickle


from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from time import time



from acrechain import load_accelerometer_csv, load_label_csv, segment_acceleration_and_calculate_features, \
    segment_labels
from VagesHAR.StatisticHelpers import generate_statistics, generate_and_save_confusion_matrix



module_root = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(module_root)
data_set_folder = os.path.join(project_root, "DATA")
feature_importances_folder = os.path.join(project_root, "FEATURE_IMPORTANCES")
models_folder = os.path.join(project_root, "MODELS")
models_dual_sensor_folder = os.path.join(models_folder, "DUAL_SENSOR")
plot_folder = os.path.join(project_root, "PLOTS")
model_accuracy_folder = os.path.join(project_root, "MODEL_ACCURACY")
statistics_folder = os.path.join(project_root, "STATISTICS")



#Config

warnings.filterwarnings('ignore')


train_overlap = 0.8
n_jobs = -1
window_length = 3.0



#List of all activities in the labeled data-set.
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
    "lie_sit": 911,
    "lie_stand": 912,
    "lie_walk": 913,
    "sit_stand": 921,
    "sit_lie" : 922,
    "sit_walk" : 923,
    "stand_lie": 931,
    "stand_sit": 932,
    "stand_walk": 933,
    "walk_lie": 941,
    "walk_sit": 942,
    "walk_stand": 943,
    "bending": 10,
    "picking": 11,
    "undefined": 12,
    "cycling": 13,
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

number_to_label_dict = dict([(label_to_number_dict[l], l) for l in label_to_number_dict])




# The activities we are relabeling
relabel_dict = {
    # 3: 9,
    4: 1,
    5: 1,
    11: 10,
    14: 13,
    20: 8,
    21: 8,
    22: 8,
    23: 8
}
keep_set = {1, 2, 6, 7, 8, 10, 13}



def find_majority_activity(window):
    counts = Counter(window)
    top = counts.most_common(1)[0][0]
    return top



#Fit the classifier to the whole data set and pickle the model




def train_model(x, y, n_estimators = 50):
    forest = RFC(n_estimators=n_estimators, class_weight="balanced")
    forest.fit(x,y)
    return forest



def train_model_and_pickle(x, y, path, n_estimators=50):
    overall_forest = RFC(n_estimators=n_estimators, class_weight="balanced")
    overall_forest.fit(x, y)

    with open(path, "wb") as f:
        pickle.dump(overall_forest, f)
    print("path: ", path)


#Pickle the feature indexes for the features with feature importances part of the top "feature_top_percentage" feature importances

def pickle_feature_indexes(features_top_percentage, feature_indexes, path):
    pickle_list = feature_indexes
    with open(path, "wb") as f:
        pickle.dump(pickle_list, f)
    print("path: ", path)



def load_features_and_labels(lb_file, th_file, label_file, raw_sampling_frequency, keep_rate, keep_transitions):
    print("Loading", lb_file, "and", th_file)
    lb_data, th_data = load_accelerometer_csv(lb_file), load_accelerometer_csv(th_file)

    shape_before_resampling = lb_data.shape

    lb_data_resampled, th_data_resampled = [], []

    if keep_rate > 1:
        print("Resampling data with window size", keep_rate)
        end_of_data = lb_data.shape[0]
        for window_start in range(0, end_of_data, keep_rate):
            window_end = min((window_start + keep_rate), end_of_data)
            average_of_lb_window = np.average(lb_data[window_start:window_end], axis=0)
            average_of_th_window = np.average(th_data[window_start:window_end], axis=0)
            lb_data_resampled.append(average_of_lb_window)
            th_data_resampled.append(average_of_th_window)

        lb_data, th_data = np.vstack(lb_data_resampled), np.vstack(th_data_resampled)
        shape_after_resampling = lb_data.shape
        print("Before resampling:", shape_before_resampling, "After resampling:", shape_after_resampling)

    resampled_sampling_frequency = raw_sampling_frequency / keep_rate

    print("Segmenting and calculating features for", lb_file, "and", th_file)
    lb_windows = segment_acceleration_and_calculate_features(lb_data, window_length=window_length,
                                                             overlap=train_overlap,
                                                             sampling_rate=resampled_sampling_frequency)
    th_windows = segment_acceleration_and_calculate_features(th_data, window_length=window_length,
                                                             overlap=train_overlap,
                                                             sampling_rate=resampled_sampling_frequency)

    features = np.hstack([lb_windows, th_windows])
    print("Number of feature returnet from ACTIVITY Chain: ", features.shape)

    print("Shape lb_windows ", lb_windows.shape)
    print("Shape th_windows ", th_windows.shape)

    #Relabel activities in the label file
    print("Loading", label_file)
    label_data = load_label_csv(label_file)
    print("Relabeling", label_file)


    for k in relabel_dict:
        np.place(label_data, label_data == k, [relabel_dict[k]])

    if (keep_transitions):
        #Introduce the five different transition types
        transitionDict = {(8, 7): 911, (8, 6): 912, (8, 1): 913, (7, 6): 921, (7, 8): 922, (7, 1): 923, (6, 8): 931,
                          (6, 7): 932, (6, 1): 933, (1, 8): 941, (1, 7): 942, (1, 6): 943}
        prevValue = -1
        value = -1
        preTransitionActivity = -1
        postTransitionActivity = -1
        transitionCounter = 0
        for i in range(len(label_data)):
            if (value == -1):
                value = label_data[i]
            else:
                prevValue = value
                value = label_data[i]
                if (value == 9):
                    if(transitionCounter == 0):
                        preTransitionActivity = prevValue
                    transitionCounter += 1
                else:
                    if(prevValue == 9):
                        postTransitionActivity = value
                        #print("preTransitionActivity ",preTransitionActivity)
                        #print("postTransitionActivity ",postTransitionActivity)
                        if((preTransitionActivity, postTransitionActivity) in transitionDict):
                            transition_type = transitionDict.get((preTransitionActivity, postTransitionActivity))
                        else:
                            transition_type = 9
                        for j in range(transitionCounter):
                            label_data[i-transitionCounter+j] = transition_type
                        transitionCounter = 0
        transitions = [911, 912, 913, 921, 922, 923, 931, 932, 933, 941, 942, 943]
        for i in transitions:
            keep_set.add(i)


    #Resample lab data
    if keep_rate > 1:
        print("Resampling label data with window size", keep_rate)
        end_of_label_data = len(label_data)

        label_data_resampled = []

        for window_start in range(0, end_of_label_data, keep_rate):
            window_end = min(window_start + keep_rate, end_of_label_data)

            label_data_resampled.append(find_majority_activity(label_data[window_start:window_end]))

        label_data = np.hstack(label_data_resampled)

        print("Before resampling:", end_of_label_data, "After resampling:", len(label_data))

    print("Segmenting", label_file)
    lab_windows = segment_labels(label_data, window_length=window_length, overlap=train_overlap,
                                 sampling_rate=resampled_sampling_frequency)
    print("Removing unwanted activities from", label_file)
    print("Lab windows", lab_windows)
    #print("Type ", type(lab_windows))
    indices_to_keep = [i for i, a in enumerate(lab_windows) if a in keep_set]
    features = features[indices_to_keep]
    lab_windows = lab_windows[indices_to_keep]
    return features, lab_windows





#Returns a list of the feature importances of the random forest created, for each each of the
def train_with_keep_rate(subject_ids, subject_files, window_length, sampling_frequency, keep_rate, keep_transitions, features_top_percentage):
    subject_windows = Parallel(n_jobs=n_jobs)(
        delayed(load_features_and_labels)(lb_file, th_file, label_file, sampling_frequency, keep_rate, keep_transitions) for
        lb_file, th_file, label_file in subject_files)


    subject_dict = dict([(s_id, windows) for s_id, windows in zip(subject_ids, subject_windows)])

    number_of_subjects = len(subject_dict)
    print("Number of subjects: ", number_of_subjects)

    feature_importances_all_subjects = []


    for s_id in subject_dict:

        test_X, test_y = subject_dict[s_id]
        len_before_deletion = len(subject_dict)
        sw_copy = subject_dict.copy()
        del sw_copy[s_id]
        len_after_deletion = len(sw_copy)
        assert len_after_deletion == len_before_deletion - 1
        train_X, train_y = zip(*sw_copy.values())
        train_X, train_y = np.vstack(train_X), np.hstack(train_y)
        my_forest = RFC(n_estimators=50, class_weight="balanced", random_state=0, n_jobs=-1)
        my_forest.fit(train_X, train_y)
        feature_importances_all_subjects.append((s_id, my_forest.feature_importances_))


        print(s_id, " Forest trained with ", my_forest.n_features_, " features")


        #print("shape train X: ", train_X.shape)
        #print("shape transformed trained X: ", transformed_X.shape)
        #print("shape test X: ", test_X.shape)
        #print("shape transformed test X: ", transformed_X_test)
        #print("shape train y: ", train_y.shape)
        #print("shape y pred: ", y_pred.shape)
        #print("shape y test; ", test_y.shape)


        #Test

        #test_X, test_y = test_X[::5], test_y[::5]
        test_X, test_y = test_X[::], test_y[::]

        a = time()
        y_pred = my_forest.predict(test_X)
        b = time()

        #print("TIME: Predict all features:", format(b - a, ".4f"), "s")
        all_features_accuracy = accuracy_score(test_y, y_pred)
        #print("y_true type ", type(test_y))
        #print("y_pred", type, type(y_pred))
        print("y_pred_all_features :", s_id, all_features_accuracy)

        if(features_top_percentage != 1):
            #Reduce feature set
            feature_importances = my_forest.feature_importances_
            indexes = getFeatureIndexes(feature_importances, features_top_percentage)
            train_X = getFeatures(train_X, indexes)
            forest_best_features = RFC(n_estimators=50, class_weight="balanced", random_state=0, n_jobs=-1).fit(train_X, train_y)
            print(s_id, " Forest trained with ", forest_best_features.n_features_, " features")
            test_X = getFeatures(test_X, indexes)
            c = time()
            y_pred_best_feature = forest_best_features.predict(test_X)
            d = time()
            #print("TIME: Predict best features:", format(d - c, ".4f"), "s")
            best_features_accuracy = accuracy_score(test_y, y_pred_best_feature)
            print("y_pred_best_features :", s_id, best_features_accuracy)


    train_X, train_y = zip(*subject_windows)
    train_X, train_y = np.vstack(train_X), np.hstack(train_y)

    return feature_importances_all_subjects


#Trains a model with the top feature_count features by feature importance
def train_with_feature_count(subject_ids, subject_files, window_length, sampling_frequency, keep_rate, keep_transitions, feature_importances, feature_count, subject_windows, subject_dict):
    #subject_windows = Parallel(n_jobs=n_jobs)(
    #    delayed(load_features_and_labels)(lb_file, th_file, label_file, sampling_frequency, keep_rate, keep_transitions) for
    #    lb_file, th_file, label_file in subject_files)


    # subject_dict = dict([(s_id, windows) for s_id, windows in zip(subject_ids, subject_windows)])



    number_of_subjects = len(subject_dict)
    print("Number of subjects: ", number_of_subjects)

    all_y_true, all_y_pred = [], []

    subject_accuracy = {}

    for s_id in subject_dict:

        test_X, test_y = subject_dict[s_id]
        len_before_deletion = len(subject_dict)
        sw_copy = subject_dict.copy()
        del sw_copy[s_id]
        len_after_deletion = len(sw_copy)
        assert len_after_deletion == len_before_deletion - 1
        train_X, train_y = zip(*sw_copy.values())
        train_X, train_y = np.vstack(train_X), np.hstack(train_y)


        # Reduce feature set

        indexes = getFeatureIndexes(feature_importances, feature_count)

        train_X = getFeatures(train_X, indexes)


        forest_feature_count = RFC(n_estimators=50, class_weight="balanced", random_state=0, n_jobs=-1).fit(train_X, train_y)
        print("Excluded subject: ", s_id, "\n Forest trained with ", forest_feature_count.n_features_, " features")






        #Test

        print("Test X before anything ", test_X.shape)
        #test_X, test_y = test_X[::5], test_y[::5]
        test_X, test_y = test_X[::], test_y[::]

        print("Test X after X[::] ",test_X.shape)
        #Reduce the number of input features
        test_X = getFeatures(test_X, indexes)

        print("Test X after getFeatures(test_X, indexes) ", test_X.shape)


        y_pred = forest_feature_count.predict(test_X)


        accuracy_subject = accuracy_score(test_y, y_pred)
        print(s_id, accuracy_subject)


        subject_accuracy[str(s_id)] = accuracy_subject

        all_y_true.append(test_y)
        all_y_pred.append(y_pred)

    all_y_true, all_y_pred = np.hstack(all_y_true), np.hstack(all_y_pred)
    accuracy = accuracy_score(all_y_true, all_y_pred)

    print("Accuracy for model with top ", feature_count, " features: ", accuracy)


    #train_X, train_y = zip(*subject_windows)
    #train_X, train_y = np.vstack(train_X), np.hstack(train_y)

    #return forest_feature_count, accuracy

    #Save confusion matrix
    generate_and_save_confusion_matrix(all_y_true, all_y_pred, number_to_label_dict,
                                       os.path.join(plot_folder, str(sampling_frequency) + "confusion_matrix_" + "feature_count_" + str(feature_count) + ".png"))

    return generate_statistics(all_y_true, all_y_pred, number_to_label_dict, feature_count, indexes), accuracy, subject_accuracy




#Returns the indexes of the features with feature importance values greater than the feature importance for the feature below the top percentage#
def getFeatureIndexes(feature_importances, feature_count):
    feature_indexes = []
    for i in range(0, feature_count):
        feature_indexes.append(feature_importances[i][0])
    print("Features indexes extracted: ", feature_indexes)
    print("Number of features ", len(feature_indexes))
    return feature_indexes



#Returns the features of a data set which corresponding to the feature indexes#
def getFeatures(data, indexes):
    #print("Data set shape before reshape: ", data.shape)
    data = data[:, indexes]
    #print("Data set shape afer reshape ", data.shape)
    return data




