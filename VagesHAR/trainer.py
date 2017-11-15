from collections import Counter

import warnings
import os
import numpy as np
import pickle

from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import accuracy_score

from acrechain import load_accelerometer_csv, load_label_csv, segment_acceleration_and_calculate_features, \
    segment_labels
from StatisticHelpers import generate_and_save_confusion_matrix

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
    "lying (right)": 23
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


def load_features_and_labels(lb_file, th_file, lab_file, raw_sampling_frequency, keep_rate):
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
    print("Loading", lab_file)
    lab_data = load_label_csv(lab_file)
    print("Relabeling", lab_file)

    for k in relabel_dict:
        np.place(lab_data, lab_data == k, [relabel_dict[k]])

    if keep_rate > 1:
        print("Resampling label data with window size", keep_rate)
        end_of_label_data = len(lab_data)

        lab_data_resampled = []

        for window_start in range(0, end_of_label_data, keep_rate):
            window_end = min(window_start + keep_rate, end_of_label_data)

            lab_data_resampled.append(find_majority_activity(lab_data[window_start:window_end]))

        lab_data = np.hstack(lab_data_resampled)

        print("Before resampling:", end_of_label_data, "After resampling:", len(lab_data))

    print("Segmenting", lab_file)
    lab_windows = segment_labels(lab_data, window_length=window_length, overlap=train_overlap,
                                 sampling_rate=resampled_sampling_frequency)
    print("Removing unwanted activities from", lab_file)
    indices_to_keep = [i for i, a in enumerate(lab_windows) if a in keep_set]
    features = features[indices_to_keep]
    lab_windows = lab_windows[indices_to_keep]
    return features, lab_windows


def train_with_keep_rate(subject_ids, subject_files, window_length, sampling_frequency, keep_rate):
    subject_windows = Parallel(n_jobs=n_jobs)(
        delayed(load_features_and_labels)(lb_file, th_file, lab_file, sampling_frequency, keep_rate) for
        lb_file, th_file, lab_file in subject_files)

    subject_dict = dict([(s_id, windows) for s_id, windows in zip(subject_ids, subject_windows)])

    all_y_true, all_y_pred = [], []

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
        test_X, test_y = test_X[::5], test_y[::5]
        y_pred = my_forest.predict(test_X)
        print(s_id, accuracy_score(test_y, y_pred))

        all_y_true.append(test_y)
        all_y_pred.append(y_pred)

    all_y_true, all_y_pred = np.hstack(all_y_true), np.hstack(all_y_pred)

    train_X, train_y = zip(*subject_windows)
    train_X, train_y = np.vstack(train_X), np.hstack(train_y)

    hz = sampling_frequency / keep_rate
    generate_and_save_confusion_matrix(all_y_true, all_y_pred, number_to_label_dict,
                                       os.path.join(project_root, str(hz) + ".png"))

    model_path = os.path.join(project_root, "healthy_" + str(window_length) + "s_model_" + str(hz) + "hz.pickle")
    train_model_and_pickle(train_X, train_y, model_path, keep_rate)


if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    window_length = 3.0
    train_overlap = 0.8
    sampling_frequency = 100
    n_jobs = -1

    project_root = os.path.dirname(os.path.abspath(__file__))
    data_set_folder = os.path.join(project_root, "tfl")
    csvs_we_are_looking_for = ["LOWERBACK", "THIGH", "labels"]

    subject_files = []

    for r, ds, fs in os.walk(data_set_folder):
        found_csvs = [False] * len(csvs_we_are_looking_for)

        for f in fs:
            for i, csv_string in enumerate(csvs_we_are_looking_for):
                if csv_string in f:
                    found_csvs[i] = os.path.join(r, f)

        if False not in found_csvs:
            subject_files.append(found_csvs)

    subject_files.sort()

    subject_ids = [os.path.basename(os.path.dirname(s)) for s, _, _ in subject_files]

    bad_performing_subjects = ["012", "014"]

    relabel_dict = {
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

    for bps_id in bad_performing_subjects:
        idx = subject_ids.index(bps_id)
        subject_ids.pop(idx)
        subject_files.pop(idx)

    print(subject_ids)
    print(subject_files)

    for keep_rate in reversed([100, 50, 25, 20, 10, 5, 4, 2]):
        print("Keep rate:", keep_rate)
        train_with_keep_rate(subject_ids, subject_files, window_length, sampling_frequency, keep_rate)
