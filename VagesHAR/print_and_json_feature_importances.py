import os
import pickle
import json
import numpy as np

module_root = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(module_root)
feature_importances_folder = os.path.join(project_root,"FEATURE_IMPORTANCES")

feature_importances_100hz = os.path.join(feature_importances_folder, "100.0hz_final_feature_importances_sorted_descending_importance.pickle")

with open(feature_importances_100hz, "rb") as f:
    feature_importances = pickle.load(f)
print(feature_importances)


feature_importances_json_path = os.path.join(feature_importances_folder, "100.0hz_final_feature_importances_sorted_descending_importance.json")



#with open(feature_importances_json_path, "w") as f:
#    json.dump(feature_importances, f, sort_keys=True, indent=4, separators=(',', ': '))


array_1 = np.array([[0,1,2], [0,2,6], [9,4,5], [1,4,6]])
print(array_1.shape)
print(array_1)
indexes = [0,2]
print(array_1[:, indexes])
