from collections import Counter

import warnings
import os
import numpy as np
import pickle
import matplotlib as plt

from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from time import time


from acrechain import load_accelerometer_csv, load_label_csv, segment_acceleration_and_calculate_features, \
    segment_labels

from VagesHAR.StatisticHelpers import generate_and_save_confusion_matrix, plot_data_distribution, generate_and_save_statistics_json



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


def find_majority_activity(window):
    counts = Counter(window)
    top = counts.most_common(1)[0][0]
    return top


def train_model_and_pickle(x, y, path, n_estimators=50):
    overall_forest = RFC(n_estimators=n_estimators, class_weight="balanced")
    overall_forest.fit(x, y)

    with open(path, "wb") as f:
        pickle.dump(overall_forest, f)
    print("path: ", path)


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
    #print("Shape lb_windows ", lb_windows.shape)
    #print("Shape th_windows ", th_windows.shape)

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


def train_with_keep_rate(subject_ids, subject_files, window_length, sampling_frequency, keep_rate, keep_transitions, features_top_percentage):
    subject_windows = Parallel(n_jobs=n_jobs)(
        delayed(load_features_and_labels)(lb_file, th_file, label_file, sampling_frequency, keep_rate, keep_transitions) for
        lb_file, th_file, label_file in subject_files)


    subject_dict = dict([(s_id, windows) for s_id, windows in zip(subject_ids, subject_windows)])

    all_y_true, all_y_pred, all_y_pred_best_features = [], [], []


    all_y_pred_accuracy_and_time = {}

    sum_accuracy_all_features = 0
    sum_accuracy_best_features= 0
    sum_test_time_all_features = 0
    sum_test_time_best_features = 0

    number_of_subjects = len(subject_dict)
    print("Number of subjects: ", number_of_subjects)
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

            all_y_pred_accuracy_and_time[s_id] = (all_features_accuracy, best_features_accuracy,
                                     b-a, d-c)
            sum_accuracy_best_features += best_features_accuracy
            sum_test_time_best_features += d - c
            all_y_pred_best_features.append(y_pred_best_feature)


        sum_accuracy_all_features += all_features_accuracy
        sum_test_time_all_features += b-a

        all_y_true.append(test_y)
        all_y_pred.append(y_pred)

    all_y_true, all_y_pred = np.hstack(all_y_true), np.hstack(all_y_pred)
    if(features_top_percentage == 1):
        average_accuracy_all_features = accuracy_score(all_y_true, all_y_pred)
        print("Keep rate: ", keep_rate)
        print("Average accuracy for all features: ", average_accuracy_all_features)
        print("Time spent on predicting with all features: ", sum_test_time_all_features)
    else:
        all_y_pred_best_features =  np.hstack(all_y_pred_best_features)
        average_accuracy_all_features = accuracy_score(all_y_true, all_y_pred)
        average_accuracy_best_features = sum_accuracy_best_features / number_of_subjects
        average_accuracy_best_features = accuracy_score(all_y_true, all_y_pred_best_features)

        for subject in all_y_pred_accuracy_and_time.items():
            print()
            print(subject[0], ": Accuracy".rjust(35), " Time:".rjust(20))
            print("All features: ", str(subject[1][0]).rjust(20), str(subject[1][2]).rjust(30))
            print("Best features: ", str(subject[1][1]).rjust(20), str(subject[1][3]).rjust(30))
            print()

        print("Keep rate: ", keep_rate)
        print("Average accuracy for all features: ", average_accuracy_all_features)
        print("Average accuracy for best features: ", average_accuracy_best_features)
        print("Difference between accuracy for best and all features (best-all): ",
              average_accuracy_best_features - average_accuracy_all_features, " Percent difference: "
              , 100 * (average_accuracy_best_features - average_accuracy_all_features))
        print("Time spent on predicting with all features: ", sum_test_time_all_features)
        print("Time spent on preditction with best features: ", sum_test_time_best_features)
        print("Time saved with best features: ", sum_test_time_all_features - sum_test_time_best_features,
              ". Percent reduction in total time: ", 1 - (sum_test_time_best_features / sum_test_time_all_features))




    train_X, train_y = zip(*subject_windows)
    train_X, train_y = np.vstack(train_X), np.hstack(train_y)



    hz = sampling_frequency / keep_rate
    if(features_top_percentage == 1):
        generate_and_save_confusion_matrix(all_y_true, all_y_pred, number_to_label_dict,
                                        os.path.join(project_root, str(hz) + ".png"))

    else:
        generate_and_save_confusion_matrix(all_y_true, all_y_pred_best_features, number_to_label_dict,
                                        os.path.join(project_root, str(hz) + "best_features" + ".png"))
        print("Best feature statistics: ")
        generate_and_save_statistics_json(all_y_true, all_y_pred_best_features, number_to_label_dict,
                                          os.path.join(project_root, "statistics" + ".json"))

    print("All features statistics: ")
    generate_and_save_statistics_json(all_y_true, all_y_pred, number_to_label_dict, os.path.join(project_root, "statistics" +".json"))

    if (features_top_percentage == 1):

        model_path = os.path.join(project_root, "healthy_" + str(window_length) + "s_model_" + str(hz) + "hz.pickle")

    else:
        train_X = getFeatures(train_X, indexes)
        model_path = os.path.join(project_root, "healthy_" + str(window_length) + "s_model_" + str(hz) + "hz_reduced.pickle")
        indexes_path = os.path.join(project_root,
                                    "indexes_" + str(features_top_percentage) + "_percent" + "_healthy_" + str(
                                        window_length) + "s_model_" + str(hz) + "hz_reduced.pickle")
        pickle_feature_indexes(features_top_percentage, indexes, indexes_path)

    train_model_and_pickle(train_X, train_y, model_path, keep_rate)





def getFeatureIndexes(feature_importances, features_top_percentage):
    number_of_features = int(features_top_percentage*len(feature_importances))
    #print("Number of features wanted: ", number_of_features)
    feature_importances_sorted = np.sort(feature_importances)
    #print("Feature importances sorted", feature_importances_sorted)
    feature_threshold = feature_importances_sorted[-number_of_features]
    #print("Feature threshold: ", feature_threshold)

    feature_indexes = []
    for i in range(len(feature_importances)):
        if len(feature_indexes) < number_of_features:
            #print("Feature importance: ", feature_importances[i], " feature threshold: ", feature_threshold)
            if feature_importances[i] >= feature_threshold:
                feature_indexes.append(i)
        else:
            break
    #print("Number of features extracted: ", len(feature_indexes))
    print("Number of features ", len(feature_indexes))
    return feature_indexes


def getFeatures(data, indexes):
    #print("Data set shape before reshape: ", data.shape)
    data = data[:, indexes]
    #print("Data set shape afer reshape ", data.shape)
    return data




if __name__ == "__main__":
    warnings.filterwarnings('ignore')


    #Trainer parameters
    window_length = 3.0
    train_overlap = 0.8
    sampling_frequency = 100
    n_jobs = -1
    keep_transitions = 1
    features_top_percentage = 0.4

    module_root = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(module_root)
    data_set_folder = os.path.join(project_root, "DATA")







    csvs_we_are_looking_for = ["LOWERBACK", "THIGH", "labels"]

    subject_files = []
    for r, ds, fs in os.walk(data_set_folder):
        found_csvs = [False] * len(csvs_we_are_looking_for)

        for f in fs:
            print("f", f)
            for i, csv_string in enumerate(csvs_we_are_looking_for):
                if csv_string in f:
                    found_csvs[i] = os.path.join(r, f)

        if False not in found_csvs:
            subject_files.append(found_csvs)



    subject_files.sort()




    subject_ids = [os.path.basename(os.path.dirname(s)) for s, _, _ in subject_files]

    print("subject ids: ", subject_ids)
    print("subject files", subject_files)

    print("label path subject 1: ", subject_files[0][2])
    labe1 = load_label_csv(subject_files[0][2])
    print("Len labe1: ", len(labe1))
    #Plot activity distribution
    #label_files = {}
    #label_data = {}
    #for i in range(len(subject_ids)):
    #    label_files[subject_ids[i]] = subject_files[i][2]
    #    label_data[subject_ids[i]] = load_label_csv(label_files[subject_ids[i]])
    #plot_data_distribution(label_data)


    #Not needed
    #bad_performing_subjects = ["001", "002", "003", "004", "005"]
    #for bps_id in bad_performing_subjects:
    #    idx = subject_ids.index(bps_id)
    #    subject_ids.pop(idx)
    #    subject_files.pop(idx)

    #print("subject_ids after removing bad performing subjects", subject_ids)
    #print("subject files after removing bad performing subjects", subject_files)


    #The activities we are relabeling
    relabel_dict = {
        #3: 9,
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



    #for keep_rate in reversed([100, 50, 25, 20, 10, 5, 4, 2]):
    for keep_rate in reversed([100]):
        print("Keep rate:", keep_rate)
        train_with_keep_rate(subject_ids, subject_files, window_length, sampling_frequency, keep_rate, keep_transitions, features_top_percentage)


