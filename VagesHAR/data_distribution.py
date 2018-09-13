import os
from acrechain import load_accelerometer_csv, load_label_csv, segment_acceleration_and_calculate_features, \
    segment_labels
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from collections import Counter




module_root = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(module_root)

#Dataset paths


data_set_folder = os.path.join(project_root, "DATA") #Trondheim Free Living (TFL) dataset
data_tcs = os.path.join(project_root, "DATA_TCS") #Trondheim Chronic Stroke (TCS dataset
data_nwt = os.path.join(project_root, "DATA_SNT") #No-wear Time
plots_folder = os.path.join(project_root, "PLOTS")

subject6path = os.path.join(data_set_folder, "006")
label6path = os.path.join(subject6path, "006_labels.csv")

print("Module root: ", module_root)
print("Project root: ", project_root)
print("'DATA' folder (TFL): ", data_set_folder)
print("'DATA_TCS' folder (TCS): ", data_tcs)
print("'DATA_SNT' folder (SNT)" , data_nwt)



label_to_number_dict = {
    "None": 0,
    "Walking": 1,
    "Running": 2,
    "Shuffling": 3,
    "Stairs (ascending)": 4,
    "Stairs (descending)": 5,
    "Standing": 6,
    "Sitting": 7,
    "Lying": 8,
    "Transition": 9,
    "Lie_sit": 911,
    "Lie_stand": 912,
    "Lie_walk": 913,
    "Sit_stand": 921,
    "Sit_lie" : 922,
    "Sit_walk" : 923,
    "Stand_lie": 931,
    "Stand_sit": 932,
    "Stand_walk": 933,
    "Walk_lie": 941,
    "Walk_sit": 942,
    "Walk_stand": 943,
    "Bending": 10,
    "Picking": 11,
    "Undefined": 12,
    "Cycling": 13,
    "Cycling (stand)": 14,
    "Heel drop": 15,
    "Vigorous activity": 16,
    "Non-vigorous activity": 17,
    "Transport(sitting)": 18,
    "Commute(standing)": 19,
    "Lying (prone)": 20,
    "Lying (supine)": 21,
    "Lying (left)": 22,
    "Lying (right)": 23,
}

number_to_label_dict = dict([(label_to_number_dict[l], l) for l in label_to_number_dict])



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
keep_set = {1, 2, 6, 7, 8, 9, 10, 13}

bad_subjects = {'001', '002', '003', '004', '005'}

#Relabel and filter out activities
def filterData(label_array, relabel_mapping = relabel_dict, keep_set = keep_set):
    #Relabe
    for k in relabel_mapping:
        np.place(label_array, label_array == k, [relabel_dict[k]])
    #Filter
    keep_indexes = [i for i, a in enumerate(label_array) if a in keep_set]
    label_array = label_array[keep_indexes]
    return label_array



def relabelTranistions(label_array):
    transitionDict = {(8,7): 911, (8,6): 912, (8,1): 913, (7,6): 921, (7,8): 922, (7,1): 923,  (6,8): 931, (6,7): 932
                      , (6,1): 933, (1,8): 941, (1,7): 942, (1,6): 943}
    prevValue = -1
    value = -1
    preTransitionActivity = -1
    postTransitionActivity = -1
    transitionCounter = 0
    for i in range(len(label_array)):
        if (value == -1):
            value = label_array[i]
        else:
            prevValue = value
            value = label_array[i]
            if (value == 9):
                if (transitionCounter == 0):
                    preTransitionActivity = prevValue
                transitionCounter += 1
            else:
                if (prevValue == 9):
                    postTransitionActivity = value
                    # print("preTransitionActivity ",preTransitionActivity)
                    # print("postTransitionActivity ",postTransitionActivity)
                    if ((preTransitionActivity, postTransitionActivity) in transitionDict):
                        transition_type = transitionDict.get((preTransitionActivity, postTransitionActivity))
                        # if((preTransitionActivity,postTransitionActivity) == (1,6)):
                        # print("                     Transition 93 occured")
                        # elif((preTransitionActivity,postTransitionActivity) == (6,7)):
                        # print("                     Transition 94 occured")
                        # elif((preTransitionActivity, postTransitionActivity) == (7,6)):
                        # print("                     Transition 91 occured")
                        # else:
                        # print("                     Transition 92 occured")
                    else:
                        transition_type = 9
                    for j in range(transitionCounter):
                        label_array[i - transitionCounter + j] = transition_type
                    transitionCounter = 0

    return label_array




csvs_we_are_looking_for = ["labels"]

label_files = []
for r, ds, fs in os.walk(data_set_folder):
    found_csvs = [False] * len(csvs_we_are_looking_for)

    for f in fs:
        print("f", f)
        for i, csv_string in enumerate(csvs_we_are_looking_for):
            if csv_string in f:
                found_csvs[i] = os.path.join(r, f)

    if False not in found_csvs:
        label_files.append(found_csvs)

label_files.sort()
subject_ids = [os.path.basename(os.path.dirname(str(s))) for s in label_files]
print("Subject IDs: ", subject_ids)


bad_performing_subjects = ["001", "002", "003", "004", "005"]
for bps_id in bad_performing_subjects:
    idx = subject_ids.index(bps_id)
    subject_ids.pop(idx)
    label_files.pop(idx)
print("Subject IDs: ", subject_ids)

print("Label_files", label_files)
labels = [load_label_csv(label_file[0]) for label_file in label_files]
filtered_labels = [filterData(label) for label in labels]
#labels_relabeld_transitions = [relabelTranistions(label) for label in filtered_labels]


#labels = [pd.read_csv(label_file[0]) for label_file in label_files]
frames = []
frames = [pd.DataFrame(label, columns=["Label"]) for label in filtered_labels]

data_frame_concatenated = pd.concat(frames, axis=0, ignore_index=True)
print(data_frame_concatenated)
samples = data_frame_concatenated.size
print("Samples: ", samples)


distribution = data_frame_concatenated.groupby("Label")["Label"].count()
#print(distribution)


distribution_dict = distribution.to_dict()
print(distribution_dict)

labeled_distribution_frequency = {}
labeled_distribution_proportion = {}

for key_value_pair in distribution_dict.items():
    labeled_distribution_frequency[number_to_label_dict[key_value_pair[0]]] = key_value_pair[1]
    labeled_distribution_proportion[number_to_label_dict[key_value_pair[0]]] = key_value_pair[1]/samples
print(labeled_distribution_proportion)








#Plots the activity distribution for one subject
def plot_individual(individual):
    index_individual = subject_ids.index(individual)
    # Individual plot
    individuals = frames
    samples = individuals[index_individual].size
    print("Number of samples for individual: ", individual, " :", samples)
    counts = individuals[index_individual].groupby("Label")["Label"].count()
    print("Counts for individual: ", individual, counts)

    distribution_dict = counts.to_dict()
    print(distribution_dict)

    labeled_distribution_frequency = {}
    labeled_distribution_proportion = {}

    for key_value_pair in distribution_dict.items():
        labeled_distribution_frequency[number_to_label_dict[key_value_pair[0]]] = key_value_pair[1]
        labeled_distribution_proportion[number_to_label_dict[key_value_pair[0]]] = key_value_pair[1] / samples
    print(labeled_distribution_proportion)
    title = "Subject " + individual +".                    Time: " + str(samples/100/60) + " minutes\n"
    title = ""
    print("Subject " + individual +".                    Time: " + str(samples/100/60) + " minutes\n")
    print()
    for activity in keep_set:
        if number_to_label_dict[activity] in labeled_distribution_proportion.keys():
            pass
        else:
            labeled_distribution_proportion[number_to_label_dict[activity]] = 0
    plot_distribution_proportion(labeled_distribution_proportion, title)



def plot_distribution_proportion(data, title):
    distribution_lists = sorted(data.items(), key=lambda dist: dist[1])
    x, y = zip(*distribution_lists)
    print(x)
    print(y)
    # plt.yscale('log')
    # plt.bar(x, y, log = True)
    fig, ax = plt.subplots()
    rects1 = ax.bar(x, y)
    plt.ylabel("Proportion", fontsize=17)
    plt.xlabel("\nActivity", fontsize=17)
    plt.xticks(rotation=90, fontsize=11)

    axes = plt.gca()
    #axes.set_ylim([0, 1])
    axes.get_yticklines()
    axes.set_yticks(np.arange(0, 1.1, 0.1))

    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = float(rect.get_height())
            print(height)

            ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                    '%0.3f' % height,
                    ha='center', va='bottom')

    autolabel(rects1)
    plt.title(title, fontsize = 14, weight = 'bold')
    plt.show()


def plot_distribution_frequency(data):
    distribution_lists = sorted(data.items(), key=lambda dist: dist[1])
    x, y = zip(*distribution_lists)
    print(x)
    print(y)
    plt.yscale('log')
    plt.bar(x, y, log = True)
    fig, ax = plt.subplots()
    rects1 = ax.bar(x, y)
    plt.ylabel("Frequency", fontsize=20)
    plt.xlabel("Activity", fontsize=20)
    plt.xticks(rotation=90, fontsize=13)

    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = float(rect.get_height())
            print(height)

            ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                    height,
                    ha='center', va='bottom')

    autolabel(rects1)


def plot_distribution_frequency_and_proportion_SNT(subjects, type_of_plot = "f"):


    window_length = 2.399999999999998
    temperature_reading_rate = 120
    sampling_frequency = 50
    samples_pr_second = 1 / (temperature_reading_rate / sampling_frequency)
    samples_pr_window = int(round(window_length * samples_pr_second))




    if len(subjects) == 1:
        sensor_config_count = {}
        protocol = subjects[0][0]
        subject_id = subjects[0][1]

        if (subject_id == 1):
            name = "atle"
        elif (subject_id == 2):
            name = "vegar"

        if (protocol == "2"):
            if (name == "atle"):
                delimitter = ";"
            else:
                delimitter = ","
        else:
            delimitter = ","

        label_path = os.path.join(data_nwt, "P" + str(protocol) + "_S" + str(subject_id) + "/" + "P" + str(
            protocol) + "_" + name + "_labels.csv")

        labels = pd.read_csv(label_path, delimiter=";", header=None).as_matrix([0])
        print(labels)
        label_windows = segment_labels(labels, samples_pr_window = samples_pr_window,  window_length=window_length, overlap=0.0)

        print(labels)
        list_configs = []
        for i in range(0, len(label_windows)):
            list_configs.append(label_windows[i].tolist()[0])
        print(list_configs)
        counter = Counter(list_configs)


        print(counter)


        save_path = os.path.join(plots_folder, "sensor_config_frequency_P" + str(protocol) + "_S" + str(subject_id) + "_samples_in_window_" + str(samples_pr_window) + "_window_length_" + str(round(window_length, 2)) + "s" + "_"+ type_of_plot + ".png")

    else:
        total_sensor_counfig_count = []
        for subject in subjects:
            print(subject)
            protocol = subject[0]
            subject_id = subject[1]

            if (subject_id == 1):
                name = "atle"
            elif (subject_id == 2):
                name = "vegar"

            label_path = os.path.join(data_nwt, "P" + str(protocol) + "_S" + str(subject_id) +"/" + "P" + str(protocol) + "_" + name + "_labels.csv")

            labels = pd.read_csv(label_path, delimiter=";", header=None).as_matrix([0])

            print(label_path)

            label_windows = segment_labels(labels, samples_pr_window = samples_pr_window,  window_length=window_length, overlap=0.0)
            print(label_windows.shape)
            this_subject_config_count = []
            for i in range(0, len(label_windows)):
                this_subject_config_count.append(label_windows[i].tolist()[0])


            total_sensor_counfig_count.append(this_subject_config_count)
        print(total_sensor_counfig_count)
        new_list = []
        for i in range(0,len(total_sensor_counfig_count)):
            for j in range(0, len(total_sensor_counfig_count[i])):
                new_list.append(total_sensor_counfig_count[i][j])
        print(new_list)
        counter = Counter(new_list)
        print(counter)




        save_path = os.path.join(plots_folder, "sensor_config_frequency_" + "P2_subjects"+ "_samples_in_window_" + str(samples_pr_window) + "_window_length_" + str(
                round(window_length, 2)) + "s" + "_" + type_of_plot + ".png")

    total_count = 0
    for value in counter.values():
        total_count += value

    print("Length of recording = ", total_count*window_length, " seconds", (total_count*window_length)/60, " minutes")

    print("Total count: ", total_count)
    y = []
    x = []
    for key, value in counter.items():

        if(key == "A"):
            label = "All (A)"
        elif(key == "B"):
            label = "Back (B)"
        else:
            label = "Thigh (T)"
        x.append(label)

        if type_of_plot == "f":
            y.append(int(value))
        elif type_of_plot =="p":
            value = value / total_count
            y.append(value)

    plt.yscale('log')
    plt.bar(x, y, log = True)
    fig, ax = plt.subplots()
    rects1 = ax.bar(x, y)
    if(type_of_plot == "f"):
        plt.ylabel("Frequency\n", fontsize=20, fontweight = "bold")
    elif(type_of_plot == "p"):
        plt.ylabel("Proportion\n", fontsize=20, fontweight = "bold")
        axes = plt.gca()
        # axes.set_ylim([0, 1])
        axes.get_yticklines()
        axes.set_yticks(np.arange(0, 1.1, 0.1))

    plt.xlabel("\n Sensor on", fontsize=20, fontweight = "bold")
    plt.xticks(rotation=0, fontsize=18)
    plt.yticks(fontsize = 18)

    fig.set_size_inches(12, 8.5)

    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            if (type_of_plot == "f"):
                height = int(rect.get_height())
                ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                        height,
                        ha='center', va='bottom', fontweight = "bold", fontsize = "17")
            elif (type_of_plot == "p"):
                height = float(rect.get_height())
                ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                        '%0.3f' % height,
                        ha='center', va='bottom', fontsize = "17")
            print(height)

    autolabel(rects1)

    fig.savefig(save_path)
    plt.clf()



def segment_labels(label_data, sampling_rate=1, window_length=120, overlap=0.0, samples_pr_window = 50):
    #window_samples = int(sampling_rate * window_length)
    window_samples = samples_pr_window
    step_size = int(round(window_samples * (1.0 - overlap)))
    print(step_size)
    labels = []

    print(label_data.shape[0], step_size)
    for window_start in np.arange(0, label_data.shape[0], step_size):
        #print(window_start, label_data.shape[0], step_size)
        window_start = int(round(window_start))
        window_end = window_start + int(round(window_samples))
        if window_end > label_data.shape[0]:
            break
        window = label_data[window_start:window_end]
        #print(window)
        top = find_majority_activity(window)
        labels.append(top)

    return np.array(labels)


def find_majority_activity(window):
    sensor_labels_list = window.tolist()
    labels_without_list = []
    for sensor_label in sensor_labels_list:
        labels_without_list.append(sensor_label[0])
    counts = Counter(labels_without_list)
    top = counts.most_common(1)[0][0]
    return top




#plot_distribution_proportion(labeled_distribution_proportion,"")
#plot_individual('022')
#fig = plot.get_figure()
#fig.savefig("plot.png")



plot_distribution_frequency_and_proportion_SNT([(1, 1), (1,2)], "p")