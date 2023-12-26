import json
import sys

with open(sys.argv[1] + '/offset0.json', 'r') as file:
    data = json.load(file)
controls_array = data.get("controls", [])
filtered_list0 = [item for item in controls_array if item != "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"]

with open(sys.argv[1] + '/offset1.json', 'r') as file:
    data = json.load(file)
controls_array = data.get("controls", [])
filtered_list1 = [item for item in controls_array if item != "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"]

with open(sys.argv[1] + '/offset2.json', 'r') as file:
    data = json.load(file)
controls_array = data.get("controls", [])
filtered_list2 = [item for item in controls_array if item != "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"]

with open(sys.argv[1] + '/offset3.json', 'r') as file:
    data = json.load(file)
controls_array = data.get("controls", [])
filtered_list3 = [item for item in controls_array if item != "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"]

with open(sys.argv[1] + '/offset4.json', 'r') as file:
    data = json.load(file)
controls_array = data.get("controls", [])
filtered_list4 = [item for item in controls_array if item != "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"]


merged_list = []
merged_list.extend(filtered_list0)
merged_list.extend(filtered_list1)
merged_list.extend(filtered_list2)
merged_list.extend(filtered_list3)
merged_list.extend(filtered_list4)

with open(sys.argv[2], 'r') as file:
    data = json.load(file)

for i, element in enumerate(merged_list):
    if i < len(data):
        data[i]["known_suffix"] = element

with open(sys.argv[3], 'w') as file:
    json.dump(data, file, indent=2)
