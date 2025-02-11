import csv
import math
import numpy as np

"""
This script processes pose and orientation data from an input text file.
It performs the following tasks:
1. Reads a text file (e.g., "latest_path_bag1.txt") containing sensor data, including orientation (as quaternions) and position values.
2. Converts the quaternion values into Euler angles (roll, pitch, yaw) using a custom transformation function.
3. Computes a sequence number for each frame based on timestamp values and a specified time range.
4. Generates an array of allowed labels by evenly spacing numbers from 0 to a specified maximum value.
5. Filters the processed results to only include those frames whose label (derived from the frame number) is in the allowed labels.
6. Writes the filtered results to an output CSV file ("output_bag.csv") with fields: label, sequence number, position (x, y, z), and Euler angles (roll, pitch, yaw).
"""

def transform(x, y, z, w):
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    roll = math.degrees(roll)
    pitch = math.degrees(pitch)
    yaw = math.degrees(yaw)

    return roll, pitch, yaw

def filter_rows_by_label(results, allowed_labels):
    filtered_results = [row for row in results if int(row['label'].split('frame')[-1].split('.png')[0]) in allowed_labels]
    return filtered_results

input_file = './latest_path_bag1.txt'
output_csv = './output_bag.csv'

#sec后四位+nsec
begin_time = 9447837609752
end_time = 9646436824056

# Specify the maximum value and the number of points
m = 886    # Maximum value
n = 400   # Number of evenly spaced points

# Generate n evenly spaced numbers from 0 to m as floats
allowed_labels_2000 = np.linspace(0, m, n)
allowed_labels=[]

for num in allowed_labels_2000:
    allowed_labels.append(int(num))
results = []
allowed_labels = list(set(allowed_labels))

n = -1
with open(input_file, 'r', encoding='utf-8') as infile:
    lines = infile.readlines()

for i, line in enumerate(lines):
    if 'orientation:' in line:
        n = n + 1
        sec = float(lines[i - 8].split(': 173828')[-1].strip())
        n_sec = float(lines[i - 7].split(':')[-1].strip())
        x = float(lines[i + 1].split(':')[-1].strip())
        y = float(lines[i + 2].split(':')[-1].strip())
        z = float(lines[i + 3].split(':')[-1].strip())
        w = float(lines[i + 4].split(':')[-1].strip())
        p_x = float(lines[i - 3].split(':')[-1].strip())
        p_y = float(lines[i - 2].split(':')[-1].strip())
        p_z = float(lines[i - 1].split(':')[-1].strip())
        roll, pitch, yaw = transform(x, y,  z, w)
        if n_sec >= 100000000:
            seq = int((((sec * 1000000000 + n_sec) - begin_time) / (end_time - begin_time) * 975))
        else:
            seq = int((((sec * 1000000000 + n_sec*10) - begin_time) / (end_time - begin_time) * 975))
        results.append({'label':'frame'+str(n) + '.png', 'seq' : seq, 'x': p_x,'y': p_y,'z': p_z,'roll': roll, 'pitch': pitch, 'yaw': yaw})
filtered_results = filter_rows_by_label(results, allowed_labels)

with open(output_csv, 'w', encoding='utf-8', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['label','seq','x','y','z','roll', 'pitch', 'yaw'])
    writer.writeheader()
    writer.writerows(filtered_results)

print(f"transform completed, saved in {output_csv}")
