import csv
import json

with open('./newgoal.json', 'r') as json_file:
    json_data = json.load(json_file)

with open('./original_goal_output.csv', 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['goal', 'target'])
    for row in json_data:
        csv_writer.writerow([row['goal'], row['target']])

with open('./similar_goal_output.csv', 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['goal', 'target'])
    for row in json_data:
        csv_writer.writerow([row['similar_goal'], row['similar_target']])

with open('./new_goal_output.csv', 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['goal', 'target'])
    for row in json_data:
        csv_writer.writerow([row['new_goal'], row['target']])
