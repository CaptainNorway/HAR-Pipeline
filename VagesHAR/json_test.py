import json
import os
import numpy as np
import pprint as pp

module_root = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(module_root)
model_accuracy_folder = os.path.join(project_root, "MODEL_ACCURACY")

statistics_path = os.path.join(model_accuracy_folder, "test.json")

def default(o):
    if isinstance(o, np.integer):
        return int(o)



d = {"test_1":np.int64(2)}
e = {"test_2":d}

#print (e)
#for key in e.keys():
#   print("Key: ", key)
#    for value in e[key].keys():
#        print(value)
#        print(type(value))
#        print(type(e[key][value]))
#        if(isinstance(e[key][value], np.int64)):
#            e[key][value] = int(e[key][value])
        #d[key][key]

#print(e)

#with open(statistics_path, "w") as f:
    #for model_feature_count in model_statistics.keys():
    #json.dump(d, f, default=default(d))
#    json.dump(e, f, sort_keys=True, indent=4, separators=(',', ': '))

#Read JSON:

json_file = open(os.path.join(model_accuracy_folder, "100.0hz_RFC_models_(1, 138, 1)_model_statistics.json"))
json_string = json_file.read()
json_data = json.loads(json_string)

#pp.pprint(json_data)


old_keys = []
new_keys = []
for key in json_data.keys():
    string = key
    if(not string == "accuracy_all_models"):
        new_string = "00" + key
        corrected_new_string = new_string[-3:]
        new_keys.append(corrected_new_string)
        old_keys.append(key)
print("old keys", old_keys)
print("new keys", new_keys)
print("number of old keys", len(old_keys))
for i in range(0, len(old_keys)):
    print(i)
    print("New", new_keys[i])
    print("Old", old_keys[i])
    if(not new_keys[i] == old_keys[i]):
        json_data[new_keys[i]] = json_data[old_keys[i]]
        del(json_data[old_keys[i]])

pp.pprint(json_data)

old_keys = []
new_keys = []
for key in json_data["accuracy_all_models"].keys():
    #print("key", key)
    new_string = "00" + key
    corrected_new_string = new_string[-3:]
    new_keys.append(corrected_new_string)
    old_keys.append(key)

for i in range(0, len(old_keys)):
    #print("New", new_keys[i])
    #print("Old", old_keys[i])
    if (not new_keys[i] == old_keys[i]):
        json_data["accuracy_all_models"][new_keys[i]] = json_data["accuracy_all_models"][old_keys[i]]
        del(json_data["accuracy_all_models"][old_keys[i]])
#print(json_data)

#print(type(json_data))
pp.pprint(json_data)


corrected_statistics_path = os.path.join(model_accuracy_folder, "100.0hz_RFC_models_(1, 138, 1)_model_statistics_corrected.json")
with open(corrected_statistics_path, "w") as f:
    json.dump(json_data, f, sort_keys=True, indent=4, separators=(',', ': '))