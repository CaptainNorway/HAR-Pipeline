import VagesHAR.trainer as trainer
import VagesHAR.basic_trainer as basic_trainer
import VagesHAR.trainer as trainer
import os
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import csv
import numpy as np
import json
from joblib import Parallel, delayed
import pprint as pp
from matplotlib.ticker import FuncFormatter

#This module is used to create all the models to be used for the different experiments in my thesis. It finds the feature importances across the
#range of subjects, it trains the models with everything from only the most important feature to all of the features. It also trains the
#models for the different sensor configurations: single and dual sensor.



module_root = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(module_root)
data_set_folder = os.path.join(project_root, "DATA")
feature_importances_folder = os.path.join(project_root, "FEATURE_IMPORTANCES")
models_folder = os.path.join(project_root, "MODELS")
models_dual_sensor_folder = os.path.join(models_folder, "DUAL_SENSOR")
plot_folder = os.path.join(project_root, "PLOTS")
model_accuracy_folder = os.path.join(project_root, "MODEL_ACCURACY")
statistics_folder = os.path.join(project_root, "STATISTICS")

#Feature importance list path (pickled):



feature_importances_paths = {
    100: os.path.join(feature_importances_folder, "100.0hz_final_feature_importances_sorted_descending_importance.pickle"),
    1: os.path.join(feature_importances_folder, "1.0hz_final_feature_importances_sorted_descending_importance.pickle"),
}

feature_importances_dict = {}

for hz in feature_importances_paths:
    with open(feature_importances_paths[hz], "rb") as f:
        feature_importances_dict[hz] = pickle.load(f)




def get_files(path):

    csvs_we_are_looking_for = ["LOWERBACK", "THIGH", "labels"]

    subject_files = []
    for r, ds, fs in os.walk(path):
        found_csvs = [False] * len(csvs_we_are_looking_for)

        for f in fs:
            print("f", f)
            for i, csv_string in enumerate(csvs_we_are_looking_for):
                if csv_string in f:
                    #path_string = os.path.join(r, f)
                    #corrected_path_string = path_string.replace(".icloud", "")
                    #second_corrected_path_string = corrected_path_string.replace("/.", "/")
                    #print("Corrected path string", corrected_path_string)
                    #print("Second corrected path string", second_corrected_path_string)
                    found_csvs[i] = os.path.join(r, f)

        if False not in found_csvs:
            subject_files.append(found_csvs)

    subject_files.sort()

    subject_ids = [os.path.basename(os.path.dirname(s)) for s, _, _ in subject_files]

    print("subject ids: ", subject_ids)
    print("subject files", subject_files)




    return subject_ids, subject_files




#Pickle feature importance for all the features and their respective feature indexes
def pickle_feature_importances_and_indexes(sorted_average_feature_importances_with_indexes, path):
    with open(path, "wb") as f:
        pickle.dump(sorted_average_feature_importances_with_indexes, f)
    print("path: ", path)


#Pickle random forest forest classifier models and their accuracy on the TFL dataset
def pickle_models(models_and_accuracies, path):
    with open(path, "wb") as f:
        pickle.dump(models_and_accuracies, f)
    print("path: ", path)




def get_averaged_feature_importance(feature_importances_all_subjects):
    number_of_subjects = len(feature_importances_all_subjects)
    number_of_features = len(feature_importances_all_subjects[0][1])
    average_feature_importances = [0]*(number_of_features)
    for excluded_subject in feature_importances_all_subjects:
        for i in range(0, number_of_features):
            average_feature_importances[i] += excluded_subject[1][i]

    for i in range(0, number_of_features):
        average_feature_importances[i] = average_feature_importances[i]/number_of_subjects

    print("Nubmer of subjects: ", number_of_subjects)
    print("Number of features", number_of_features)
    return average_feature_importances, number_of_features


#Sort the avergage_features_list by the feature importance in descending order, and keep track of
def get_sorted_averaged_features(average_feature_importances, number_of_features):
    #Add indexes
    averaged_feature_importances_with_indexes = []
    for i in  range(0, number_of_features):
        averaged_feature_importances_with_indexes.append((i, average_feature_importances[i]))
    sorted_averaged_feature_importances_with_indexes = sorted(averaged_feature_importances_with_indexes, key=lambda tup: tup[1], reverse= True)
    return sorted_averaged_feature_importances_with_indexes



#Plotting#

#Plots the feature importances of both sensors combined
def plot_feature_importances_combined(feature_importances, frequency, sorted = False):
    path = os.path.join(plot_folder, str(frequency) + "hz_both_sensors_feature_importances_TFL_data_set_sorted_by_index.png")
    number_of_features = len(feature_importances)
    print(matplotlib.matplotlib_fname())

    indexes = []
    feature_importances_removed_indexes = []
    for i in range(0, number_of_features):
        indexes.append(feature_importances[i][0])
        feature_importances_removed_indexes.append(feature_importances[i][1])
    x = [i for i in range(1, number_of_features+1)]
    print(x)
    print(indexes)
    print(feature_importances_removed_indexes)
    plt.plot(indexes, feature_importances_removed_indexes, 'ro')
    plt.axis([1, 138, 0, feature_importances[0][1]+0.005])
    plt.xticks(x)
    plt.xticks(rotation=90, fontsize=7)
    plt.ylabel("Feature importance", fontsize=17)
    plt.xlabel("\nFeatures (most important at index 1)", fontsize=17)
    plt.title('Averaged feature importance', fontsize = 25)
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(25, 8)
    plt.savefig(path)
    plt.show()
    plt.clf()

def plot_feature_importances_seperate(feature_importances, sensor, frequency, on_top = False):

    number_of_features = int(len(feature_importances)/2)

    feature_index_range = (0,0)

    if(sensor == "lower_back"):
        feature_index_range = (0,68)
    elif(sensor == "thigh"):
        feature_index_range = (69,137)


    if (on_top):
        path = os.path.join(plot_folder,
                            str(frequency) + "hz_" + "both_sensor_overlayed" + "_feature_importances_TFL_data_set.png")
        lower_back_index_range = (0,68)
        thigh_index_range = (69, 137)

        lower_back_features = []
        thigh_features = []

        for feature in feature_importances:
            if (feature[0]>= lower_back_index_range[0] and feature[0] <= lower_back_index_range[1]):
             lower_back_features.append((feature[0]%69, feature[1]))
            else:
                thigh_features.append((feature[0]%69, feature[1]))

        lower_back_indexes, lower_back_feature_importances_removed_indexes, thigh_indexes, thigh_feature_importances_removed_indexes = [], [], [], []

        for i in range(0, number_of_features):
            lower_back_indexes.append(lower_back_features[i][0])
            lower_back_feature_importances_removed_indexes.append(lower_back_features[i][1])
            thigh_indexes.append(thigh_features[i][0])
            thigh_feature_importances_removed_indexes.append(thigh_features[i][1])

        x = [i for i in range(1, number_of_features + 1)]

        plt.plot(x, lower_back_feature_importances_removed_indexes, 'ro', x, thigh_feature_importances_removed_indexes,'bo')
        plt.axis([1, number_of_features, 0, feature_importances[0][1] + 0.005])
        plt.xticks(x)
        plt.xticks(rotation=90, fontsize=16)
        plt.yticks(fontsize = 16)
        plt.ylabel("Feature importance\n", fontsize=24)
        plt.xlabel("\nFeatures", fontsize=24)
        #plt.title('Averaged feature importance: ' + "lb and th\n", fontsize=25)
        plt.grid(True)
        plt.tight_layout()
        red_patch = mpatches.Patch(color='red', label='lb sensor')
        blue_patch = mpatches.Patch(color='blue', label ='th sensor')
        plt.rcParams["legend.fontsize"] = 22
        plt.legend(handles=[red_patch, blue_patch])
        fig = matplotlib.pyplot.gcf()
        #plt.rcParams["figure.figsize"] = [60,20]
        fig.set_size_inches(22, 8)
        plt.subplots_adjust(left = 0.14, bottom = 0.2)
        plt.savefig(path)
        plt.clf()





    else:
        path = os.path.join(plot_folder,
                            str(frequency) + "hz_" + sensor + "_sensor_feature_importances_TFL_data_set.png")
        print("Feature_index_rage: ", feature_index_range)
        sensor_features = []
        for feature in feature_importances:
            if (feature[0]>= feature_index_range[0] and feature[0] <= feature_index_range[1]):
             sensor_features.append((feature[0]%69, feature[1]))

        print("sensor features ", sensor_features)
        indexes = []
        feature_importances_removed_indexes = []
        for i in range(0, number_of_features):
            indexes.append(sensor_features[i][0])
            feature_importances_removed_indexes.append(sensor_features[i][1])

        x = [i for i in range(1, number_of_features + 1)]

        print(x)
        print(indexes)
        print(feature_importances_removed_indexes)
        print("len feature_importances_removed_indexes", len(feature_importances_removed_indexes))
        if (sensor == "lower_back"):
            color = "ro"
        else:
            color = "bo"
        plt.plot(x, feature_importances_removed_indexes, color)
        plt.axis([1, number_of_features, 0, feature_importances[0][1] + 0.005])
        plt.xticks(x)
        plt.xticks(rotation=90, fontsize=16)
        plt.yticks(fontsize = 16)
        plt.ylabel("Feature importance\n", fontsize=24)
        plt.xlabel("\nFeatures", fontsize=24)
        if (sensor == "lower_back"):
            red_patch = mpatches.Patch(color='red', label='lb sensor')
            plt.rcParams["legend.fontsize"] = 22
            plt.legend(handles=[red_patch])
        else:
            blue_patch = mpatches.Patch(color='blue', label='th sensor')
            plt.rcParams["legend.fontsize"] = 22
            plt.legend(handles=[blue_patch])
        #plt.title('Averaged feature importance: ' + sensor, fontsize=25)
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(22, 8)
        plt.grid(True)
        plt.subplots_adjust(bottom=0.2)
        plt.savefig(path)
        plt.clf()


def plot_all_time_statistics(time_statistics_path):
    json_file = open(time_statistics_path)
    json_string = json_file.read()
    json_data = json.loads(json_string)

    pp.pprint(json_data)

    print(type(json_data))

    list_feature_time_tuples = []
    for key, item in json_data.items():
        list_feature_time_tuples.append((key, item))
    print(list_feature_time_tuples)

    list_train_model_time = []
    list_extracting_windows_from_csv_time = []
    list_feature_calculate_features_time = []
    list_prediction_time = []



    for item in list_feature_time_tuples:
        list_train_model_time.append((item[0], item[1]["1.Time to train model"]))
        list_extracting_windows_from_csv_time.append((item[0], item[1]["2.Prediction pipeline"]["1.Extracting windows from CSVs"]))
        list_feature_calculate_features_time.append((item[0], item[1]["2.Prediction pipeline"]["2.Calculate features for all windows"]))
        list_prediction_time.append((item[0], item[1]["2.Prediction pipeline"]["3.Predicting"]))


    # Feature count starting with the one with the highest feature importance, then adding
    x = []
    # The feature calculation time
    y_train_model = []
    y_extract_windows = []
    y_calculate_features = []
    y_prediction = []
    for i in range(len(list_train_model_time)):
        x.append(int(list_train_model_time[i][0][:3]))
        y_train_model.append(float(list_train_model_time[i][1]))
        y_extract_windows.append(float(list_extracting_windows_from_csv_time[i][1]))
        y_calculate_features.append(float(list_feature_calculate_features_time[i][1]))
        y_prediction.append(float(list_prediction_time[i][1]))


    print(x)
    print(y_train_model)
    print(y_extract_windows)
    print(y_calculate_features)
    print(y_prediction)


    start_feature = 1
    end_feature = 138
    color = "black"


    #plt.plot(x, y_train_model,"b-", label = "Train model")
    #plt.plot(x, y_extract_windows, "k-", label = "Extract windows from CSVs")
    #plt.plot(x, y_calculate_features, "r-", label = "Calculate features")
    #plt.plot(x, y_prediction, "g-", label = "Prediction")


    plt.plot(x, y_train_model,"b-")
    plt.plot(x, y_extract_windows, "k-")
    plt.plot(x, y_calculate_features, "r-")
    plt.plot(x, y_prediction, "g-")

    red_patch = mpatches.Patch(color='red', label= "Feature calculation")
    blue_patch = mpatches.Patch(color='blue', label= "Model training")
    green_patch = mpatches.Patch(color='green', label= "Prediction")
    black_patch = mpatches.Patch(color = "black", label = "Window extraction")
    plt.rcParams["legend.fontsize"] = 15
    plt.legend(handles=[red_patch, blue_patch, green_patch, black_patch])

    plt.axis([start_feature, end_feature, 0, 25])
    plt.xticks(np.arange(1, end_feature+1, step=5))
    plt.xticks(rotation=0, fontsize=10, fontweight = "bold")
    plt.yticks(fontsize=10, fontweight = "bold", color = color)
    plt.ylabel("Time (s)\n", fontsize=16, fontweight = "bold", color = color)
    plt.xlabel("\n Features (#)", fontsize=16, fontweight = "bold")
    #plt.title('Feature calculation\n ', fontsize=18, fontweight = "bold")
    plt.grid(True)
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(12, 8)

    save_path = os.path.join(plot_folder, "100hz_time_statistics_all_pipeline_steps_" + str(start_feature) +"_" + str(end_feature) + "_1_" +"_color_" + color + "_.png")



    plt.savefig(save_path)
    plt.clf()

def plot_time_statistics(time_statistics_path):
    json_file = open(time_statistics_path)
    json_string = json_file.read()
    json_data = json.loads(json_string)

    pp.pprint(json_data)

    print(type(json_data))

    list_feature_time_tuples = []
    for key, item in json_data.items():
        list_feature_time_tuples.append((key, item))
    print(list_feature_time_tuples)
    list_feature_calculate_features_time = []
    for item in list_feature_time_tuples:
        list_feature_calculate_features_time.append((item[0], item[1]["2.Prediction pipeline"]["2.Calculate features for all windows"]))
    print(list_feature_calculate_features_time)
    # Feature count starting with the one with the highest feature importance, then adding
    x = []
    # The feature calculation time
    y = []
    for item in list_feature_calculate_features_time:
        x.append(int(item[0][:3]))
        y.append(float(item[1]))
    print(x)
    print(y)


    start_feature = 15
    end_feature = 50
    color = "black"


    plt.plot(x, y, "r-")
    plt.axis([start_feature, end_feature, y[start_feature-1]-(y[start_feature-1]*0.10), y[end_feature] + (y[end_feature-1]*0.10)])
    plt.xticks(np.arange(1, end_feature+1, step=1))
    plt.xticks(rotation=0, fontsize=10, fontweight = "bold")
    plt.yticks(fontsize=10, fontweight = "bold", color = color)
    plt.ylabel("Time (s)\n", fontsize=16, fontweight = "bold", color = color)
    plt.xlabel("\n Features (#)", fontsize=16, fontweight = "bold")
    #plt.title('Feature calculation\n ', fontsize=18, fontweight = "bold")
    plt.grid(True)
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(12, 8)

    save_path = os.path.join(plot_folder, "100hz_time_statistics_feature_calculation_features(" + str(start_feature) +"," + str(end_feature) + ",1)" +"_color_" + color + "_.png")



    plt.savefig(save_path)
    plt.clf()




def plot_accuracy(accuracy_path):
    json_file = open(accuracy_path)
    json_string = json_file.read()
    json_data = json.loads(json_string)

    pp.pprint(json_data)

    print(type(json_data))

    list_feature_statistics_tuples = []
    for key, item in json_data.items():
        list_feature_statistics_tuples.append((key, item))
    print(list_feature_statistics_tuples)
    list_feature_accuracy_tuples = []
    for item in list_feature_statistics_tuples[:-1]:
        print(item[1])
        list_feature_accuracy_tuples.append((item[0], item[1]["accuracy"]))
    print(list_feature_accuracy_tuples)
    # Feature count starting with the one with the highest feature importance, then adding
    x = []
    # The feature calculation time
    y = []
    for item in list_feature_accuracy_tuples:
        x.append(int(item[0][:3]))
        y.append(float(item[1]))
    print(x)
    print(y)




    start_feature = 1
    end_feature = 10
    number_of_features_to_show = len(x)-(-end_feature + 138)
    print("Number of features to show: ", number_of_features_to_show)
    plt.plot(x, y , "b")
    plt.axis([start_feature, number_of_features_to_show, 0.45, 1])
    print(len(x))
    plt.xticks(np.arange(start_feature, number_of_features_to_show+1, step=1))
    ax = plt.gca()
    plt.xticks(rotation=90, fontsize=14)
    plt.yticks(fontsize = 14)
    plt.ylabel("Accuracy (%)\n", fontsize=15, fontweight = "bold")
    plt.xlabel("\nFeatures (#)", fontsize=15, fontweight = "bold")
    #plt.title('Accuracy accross all subjects\n ', fontsize=17,  fontweight='bold')

    formatter = FuncFormatter(to_percent)
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.hlines(0.90, start_feature, end_feature)
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(10, 7)
    plt.subplots_adjust(left = 0.2, bottom=0.2)
    plt.grid(True)



    save_path = os.path.join(plot_folder, "100hz_accuracy_statistics_features(" + str(start_feature) + "," + str(number_of_features_to_show) + ",1).png")

    plt.savefig(save_path)
    plt.clf()

def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(100 * y)

    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'


def plot_accuracy_and_time(accuracy_statistics_path, time_statistics_path):
    plt.clf()
    json_file_accuracy = open(accuracy_statistics_path)
    json_string_accuracy = json_file_accuracy.read()
    json_data_accuracy = json.loads(json_string_accuracy)

    json_file_time = open(time_statistics_path)
    json_string_time = json_file_time.read()
    json_data_time = json.loads(json_string_time)


    # Extract the accuracy statistics fromm the JSON
    list_feature_statistics_tuples = []
    for key, item in json_data_accuracy.items():
        list_feature_statistics_tuples.append((key, item))
   # print(list_feature_statistics_tuples)
    list_feature_accuracy_tuples = []
    for item in list_feature_statistics_tuples[:-1]:
        #print(item[1])
        list_feature_accuracy_tuples.append((item[0], item[1]["accuracy"]))
    #print(list_feature_accuracy_tuples)
    # Feature count starting with the one with the highest feature importance, then adding
    x_accuracy = []
    # The feature calculation time
    y_accuracy = []
    for item in list_feature_accuracy_tuples:
        x_accuracy.append(int(item[0][:3]))
        y_accuracy.append(float(item[1]))


    print(x_accuracy)
    print(y_accuracy)

    # Extract the time statistics from the JSON

    list_feature_time_tuples = []
    for key, item in json_data_time.items():
        list_feature_time_tuples.append((key, item))
    #print(list_feature_time_tuples)
    list_feature_calculate_features_time = []
    for item in list_feature_time_tuples:
        list_feature_calculate_features_time.append(
            (item[0], item[1]["2.Prediction pipeline"]["2.Calculate features for all windows"]))
    #print(list_feature_calculate_features_time)
    # Feature count starting with the one with the highest feature importance, then adding
    x_time = []
    # The feature calculation time
    y_time = []
    for item in list_feature_calculate_features_time:
        x_time.append(int(item[0][:3]))
        y_time.append(float(item[1]))

    print(x_time)
    print(y_time)

    # Plot settings




    fig, ax1 = plt.subplots(figsize=(21, 8), dpi = 120)
    ax1.plot(x_accuracy, y_accuracy, 'ob-')
    ax1.set_ylim(0.9,0.95)

    # Make the y-axis label, ticks and tick labels match the line color .
    ax1.set_ylabel('Accuracy (%)\n', color = "b", fontsize=22, fontweight = "bold")
    ax1.tick_params('y', colors='b', labelsize = 20.0)
    ax1.tick_params('x', rotation=90, labelsize = 20.0)

    ax2 = ax1.twinx()
    ax2.plot(x_accuracy, y_time, "or-")
    ax2.set_ylabel('\n Feature calculation time (s)', color = "r", fontsize=22, fontweight = "bold")
    ax2.tick_params('y', colors = "r", labelsize = 20.0)
    ax2.tick_params('x', rotation = 90, labelsize = 20.0)

    ax1.set_xlabel("\n Features (#)", fontsize=22, fontweight = "bold")

    xmin, xmax = plt.xlim()  # return the current xlim
    plt.xlim((15, 50))

    plt.ylim((2,10))


    plt.xticks(np.arange(15, 51 , step=1))
    formatter = FuncFormatter(to_percent)
    ax1.yaxis.set_major_formatter(formatter)

    plt.subplots_adjust(top=0.96, bottom = 0.2)
    #fig.set_size_inches(20, 10)
    plt.grid(True)

    plt.hlines(8.4, 15, 50)
    #plt.title("Accuracy and feature calculation time\n", fontsize=22,  fontweight='bold')

    save_path = os.path.join(plot_folder, "100hz_accuracy_and_time_statistics_features(15,50,1).png")

    plt.savefig(save_path)
    plt.clf()


def train_models_with_feature_count(subject_ids, subject_files, window_length, sampling_frequency, keep_rate, keep_transitions, feature_importances, models_to_create, n_jobs):
    model_statistics = {}

    print("Models to create: ", models_to_create)

    start = models_to_create[0]
    end = models_to_create[1]
    step = models_to_create[2]

    feature_counts = [count for count in range(start, end + 1, step)]

    print("Feature_counts ", feature_counts)

    subject_windows = Parallel(n_jobs=n_jobs)(
        delayed(basic_trainer.load_features_and_labels)(lb_file, th_file, label_file, sampling_frequency, keep_rate, keep_transitions)
        for
        lb_file, th_file, label_file in subject_files)

    subject_dict = dict([(s_id, windows) for s_id, windows in zip(subject_ids, subject_windows)])

    accuracy_all_models = {}
    for feature_count in feature_counts:
        print("Feature count: ", feature_count)
        statistics, accuracy, subject_accuracy = basic_trainer.train_with_feature_count(subject_ids, subject_files, window_length, sampling_frequency, keep_rate, keep_transitions, feature_importances, feature_count, subject_windows, subject_dict)
        statistics["accuracy_for_all_subjects"] = subject_accuracy
        feature_count_string = "00" + str(feature_count)
        model_statistics[feature_count_string[-3:]] = statistics
        accuracy_all_models[feature_count_string[-3:]]= accuracy
    model_statistics["accuracy_all_models"] = accuracy_all_models
    return model_statistics


    #for i in range(0, number_of_models):
    #    feature_count = number_of_features - (number_of_features - i) + 1
    #    print("Feature_count: ", feature_count)
    #    model = basic_trainer.train_with_feature_count(subject_ids, subject_files, window_length, sampling_frequency, keep_rate, keep_transitions, feature_importances, feature_count)
    #    models.append((feature_count, model))

    #return models



#Run once: Find the correct feature importances


#Run config:
window_length = 3.0
sampling_frequency = 100
keep_transitions = 0
n_jobs = -1

features_top_percentage = 1

keep_rate = 1



subject_ids, subject_files = get_files(data_set_folder)
#print(subject_files)
print("Keep rate: Every ", keep_rate, "th sample, hz = 100/keep_rate = ", 100 / keep_rate)
real_sampling_frequency = 100/keep_rate
print("Real sampling frequency: ", real_sampling_frequency)

for hz in feature_importances_dict.keys():
    print(str(hz) + " hz feature importances ")
    print(feature_importances_dict[hz])
    print("")


#plot_feature_importances_seperate(feature_importances_dict[100], "lower_back", 100, False)
#plot_feature_importances_seperate(feature_importances_dict[100], "thigh", 100, False)
#plot_feature_importances_seperate(feature_importances_dict[100], "lower_back", 100, True)
#plot_feature_importances_combined(feature_importances_dict[100], 100)



#Train n models starting with a model with the most importan feature then keep adding features up until the feature count ecuals n


start = 138
end = 138
step = 1
models_to_create = (start, end, step)



#print(feature_importances)
print(models_to_create)
#model_statistics =  train_models_with_feature_count(subject_ids, subject_files, window_length, sampling_frequency, keep_rate, keep_transitions, feature_importances_dict[real_sampling_frequency], models_to_create, n_jobs)
#print("Models", models)

#print(model_statistics)


#statistics_path = os.path.join(model_accuracy_folder, str(real_sampling_frequency) + "hz_RFC_models_" + "(" + str(start) + ", " + str(end) +", " + str(step) + ")_model_statistics.json")




time_statistics_path = os.path.join(statistics_folder, "100.0hz_time_statistics_training_and_predicting_(1,138,1).json")
accuracy_path = os.path.join(model_accuracy_folder, "100.0hz_RFC_models_(1, 138, 1)_model_statistics_corrected.json")

#plot_time_statistics(time_statistics_path)
#plot_accuracy(accuracy_path)
plot_accuracy_and_time(accuracy_path,time_statistics_path)
#plot_all_time_statistics(time_statistics_path)




#d = {"test_1":1}
#e = {"test_2": d}
#with open(statistics_path, "w") as f:
    #for model_feature_count in model_statistics.keys():
#    json.dump(model_statistics, f, sort_keys=True, indent=4, separators=(',', ': '))
    #json.dump(e, f)


#pickle_models(models, os.path.join(models_dual_sensor_folder, str(real_sampling_frequency) + "hz_RFC_models_" + "(" + str(start) + ", " + str(end) +", " + str(step) + ").pickle"))





#Run once: Find the correct feature importances

#feature_importances_all_subjects = basic_trainer.train_with_keep_rate(subject_ids, subject_files, window_length, sampling_frequency, keep_rate, keep_transitions, features_top_percentage)

#print (feature_importances_all_subjects)
#for subject in feature_importances_all_subjects:
#    print("Subject excluded: ", subject[0])
#    print(subject[1])

#average_feature_importances, number_of_features = get_averaged_feature_importance(feature_importances_all_subjects)
#print("Average feature importances: ", average_feature_importances)
#sorted_average_feature_importances_with_indexes = get_sorted_averaged_features(average_feature_importances, number_of_features)
#print("Average feature importances sorted in descending order, with the feature index at 0 index and importance at 1 index: ", sorted_average_feature_importances_with_indexes)

#Done once:
#pickle_feature_importances_and_indexes(sorted_average_feature_importances_with_indexes, os.path.join(feature_importances_folder, str(real_sampling_frequency) + "hz_final_feature_importances_sorted_descending_importance.pickle"))

