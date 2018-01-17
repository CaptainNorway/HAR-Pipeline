import os
from acrechain import load_accelerometer_csv, load_label_csv, segment_acceleration_and_calculate_features, \
    segment_labels
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties



module_root = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(module_root)
data_set_folder = os.path.join(project_root, "DATA")

subject6path = os.path.join(data_set_folder, "006")
label6path = os.path.join(subject6path, "006_labels.csv")

print("Module root: ", module_root)
print("Project root: ", project_root)
print("Data set folder: ", data_set_folder)


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


plot_distribution_proportion(labeled_distribution_proportion,"")

#plot_individual('022')


#fig = plot.get_figure()
#fig.savefig("plot.png")
