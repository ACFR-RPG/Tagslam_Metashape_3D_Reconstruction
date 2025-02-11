import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
# Import 3D plotting toolkit (optional)
from mpl_toolkits.mplot3d import Axes3D

"""
This Python script performs the following tasks:
1. Reads local coordinate data and estimated coordinate data from tagslam ("output_bag1.txt").
2. Computes the rigid-body transformation (rotation matrix and translation vector) that best aligns the original coordinates with the estimated coordinates.
3. Applies the transformation to the original coordinates and calculates the Euclidean errors relative to the estimated coordinates.
4. Displays a histogram of the error distribution.
5. Creates a results table and identifies the top 5 points with the highest errors.
6. Visualizes the transformed, estimated, and high-error points in 3D, including connecting lines between the transformed and estimated high-error points.
7. Compares the 3D trajectories of the transformed and estimated positions.
8. Processes Euler angle data (Yaw, Pitch, Roll) and plots their time series.
9. Plots histograms of the Euler angle errors.
10. Converts Euler angles to quaternions, plots their time evolution, and shows histograms of the quaternion errors.
11. Plots the 3D trajectory of the estimated positions.

Note:
- The input file ("output_bag1.txt") should have two header/comment lines. The first line is a comment and the second line contains column names (with the first column starting with a '#' symbol).
- The script skips the first line and processes the second line as header.
"""

# ---------------------------
# 1. Read Data and Preprocessing
# ---------------------------
# Skip the first line (comment) and use the second line as header
data = pd.read_csv('output_bag1.txt', delimiter='\t', skiprows=1)

# Remove any '#' prefix and extra whitespace from the column names
data.columns = [col.strip().lstrip('#') for col in data.columns]
# Remove the last row (similar to MATLAB's data = data(1:end-1, :))
data = data.iloc[:-1, :]
print(data.columns)

# Extract relevant columns from the DataFrame (assuming column names are as in MATLAB)
X      = data['X'].to_numpy()
Y      = data['Y'].to_numpy()
Z      = data['Z'].to_numpy()
X_est  = data['X_est'].to_numpy()
Y_est  = data['Y_est'].to_numpy()
Z_est  = data['Z_est'].to_numpy()

# Combine original coordinates and estimated coordinates into arrays
XYZ     = np.column_stack((X, Y, Z))
XYZ_est = np.column_stack((X_est, Y_est, Z_est))

# ---------------------------
# 2. Compute Rotation Matrix and Translation Vector
# ---------------------------
# Compute the centroids of both sets of coordinates
centroid_XYZ     = np.mean(XYZ, axis=0)
centroid_XYZ_est = np.mean(XYZ_est, axis=0)

# Center the data by subtracting the centroids
XYZ_centered     = XYZ - centroid_XYZ
XYZ_est_centered = XYZ_est - centroid_XYZ_est

# Compute the covariance matrix H
H = XYZ_centered.T @ XYZ_est_centered

# Perform Singular Value Decomposition (SVD) on H
U, S, Vt = np.linalg.svd(H)
R_mat = Vt.T @ U.T

# Check if a reflection is needed (ensure R_mat is a proper rotation matrix with determinant = 1)
if np.linalg.det(R_mat) < 0:
    Vt[-1, :] = -Vt[-1, :]
    R_mat = Vt.T @ U.T

# Compute the translation vector T
T = centroid_XYZ_est - R_mat @ centroid_XYZ

# Apply the rotation and translation to transform the original coordinates
XYZ_transformed = (R_mat @ XYZ.T).T + T

# Compute the Euclidean error between the transformed coordinates and the estimated coordinates
adjusted_errors = np.sqrt(np.sum((XYZ_transformed - XYZ_est)**2, axis=1))

# Plot the histogram of the adjusted errors
plt.figure(1)
plt.hist(adjusted_errors, bins=20, edgecolor='black')
plt.title('Error Distribution (XYZ Transformed vs Estimated)', fontsize=28)
plt.xlabel('Error (m)', fontsize=28)
plt.ylabel('Frequency', fontsize=28)
plt.grid(True)

# ---------------------------
# 3. Create Results Table and Identify Top 5 Error Points
# ---------------------------
# Create a DataFrame that includes the original data, estimated data, and the computed error
results_table = pd.DataFrame({
    'First_Column': data.iloc[:, 0],
    'X': X,
    'Y': Y,
    'Z': Z,
    'X_est': X_est,
    'Y_est': Y_est,
    'Z_est': Z_est,
    'Adjusted_Error': adjusted_errors
})

# Find the indices of the top 5 points with the highest errors
max_idx = np.argsort(adjusted_errors)[-5:]
max_error_data = results_table.iloc[max_idx].copy()

# Replace the original coordinates for the high-error points with the transformed coordinates
max_error_data.loc[:, 'X'] = XYZ_transformed[max_idx, 0]
max_error_data.loc[:, 'Y'] = XYZ_transformed[max_idx, 1]
max_error_data.loc[:, 'Z'] = XYZ_transformed[max_idx, 2]

print('Top 5 points with the highest adjusted errors:')
print(max_error_data)

# ---------------------------
# 4. 3D Visualization of Transformed, Estimated, and High-Error Points
# ---------------------------
fig6 = plt.figure()
ax6 = fig6.add_subplot(111, projection='3d')
# Plot the transformed points (red)
ax6.scatter(XYZ_transformed[:, 0], XYZ_transformed[:, 1], XYZ_transformed[:, 2],
            s=40, c='r', marker='o', label='Transformed')
# Plot the estimated points (blue)
ax6.scatter(XYZ_est[:, 0], XYZ_est[:, 1], XYZ_est[:, 2],
            s=40, c='b', marker='o', label='Estimated')
# Plot the high-error points (transformed, in purple with a black edge)
ax6.scatter(max_error_data['X'], max_error_data['Y'], max_error_data['Z'],
            s=100, c='m', marker='p', label='Top 5 Error (Transformed)', edgecolors='k', linewidths=2)
# Plot the estimated coordinates of the high-error points (green triangles)
ax6.scatter(max_error_data['X_est'], max_error_data['Y_est'], max_error_data['Z_est'],
            s=100, c='g', marker='^', label='Top 5 Error (Estimated)', edgecolors='k', linewidths=2)

# Draw lines connecting the transformed and estimated high-error points
for i in range(len(max_error_data)):
    ax6.plot([max_error_data['X'].iloc[i], max_error_data['X_est'].iloc[i]],
             [max_error_data['Y'].iloc[i], max_error_data['Y_est'].iloc[i]],
             [max_error_data['Z'].iloc[i], max_error_data['Z_est'].iloc[i]],
             'k-', linewidth=2)

ax6.set_title('3D Visualization of Original, Estimated, and High-Error Points', fontsize=20)
ax6.set_xlabel('X', fontsize=16)
ax6.set_ylabel('Y', fontsize=16)
ax6.set_zlabel('Z', fontsize=16)
ax6.legend(fontsize=12)
ax6.grid(True)

# ---------------------------
# 5. 3D Trajectory Comparison (Transformed vs Estimated) - Corresponds to MATLAB Figure(2)
# ---------------------------
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.plot(XYZ_transformed[:, 0], XYZ_transformed[:, 1], XYZ_transformed[:, 2],
         'r-', linewidth=2, label='Tagslam')
ax2.plot(XYZ_est[:, 0], XYZ_est[:, 1], XYZ_est[:, 2],
         'b-', linewidth=2, label='Estimated')
ax2.scatter(XYZ_transformed[:, 0], XYZ_transformed[:, 1], XYZ_transformed[:, 2],
            s=40, c='r', marker='o')
ax2.scatter(XYZ_est[:, 0], XYZ_est[:, 1], XYZ_est[:, 2],
            s=40, c='b', marker='o')
ax2.set_xlabel('X', fontsize=14)
ax2.set_ylabel('Y', fontsize=14)
ax2.set_zlabel('Z', fontsize=14)
ax2.legend(fontsize=14)
ax2.set_title('Comparison of Transformed and Estimated Positions', fontsize=16)
ax2.grid(True)

# ---------------------------
# 6. Process Euler Angle Data and Plot Time Series
# ---------------------------
# Extract Euler angle data
Yaw      = data['Yaw'].to_numpy()
Pitch    = data['Pitch'].to_numpy()
Roll     = data['Roll'].to_numpy()
Yaw_est  = data['Yaw_est'].to_numpy()
Pitch_est= data['Pitch_est'].to_numpy()
Roll_est = data['Roll_est'].to_numpy()

eul_angles     = np.column_stack((Yaw, Pitch, Roll))
eul_est_angles = np.column_stack((Yaw_est, Pitch_est, Roll_est))

# Assume the data points are evenly distributed over 120 seconds
N = eul_angles.shape[0]
t = np.linspace(0, 120, N)

# Plot Euler angles over time (corresponding to MATLAB Figure(3))
titles_eul = ['Yaw (°)', 'Pitch (°)', 'Roll (°)']
fig3, axs3 = plt.subplots(3, 1, figsize=(10, 15))
for i in range(3):
    axs3[i].plot(t, eul_angles[:, i], color='r', linewidth=2, label='Tagslam')
    axs3[i].plot(t, eul_est_angles[:, i], color='k', linestyle='--', linewidth=2, label='Estimate')
    axs3[i].set_title(titles_eul[i], fontsize=14, fontweight='bold')
    axs3[i].set_xlabel('Time (s)')
    axs3[i].set_ylabel(titles_eul[i])
    axs3[i].legend(fontsize=12, loc='best')
    axs3[i].grid(True)
    axs3[i].tick_params(labelsize=24)

# ---------------------------
# 7. Plot Histogram of Euler Angle Errors (Corresponding to MATLAB Figure(4))
# ---------------------------
error_eul = eul_angles - eul_est_angles
titles_error_eul = ['Yaw Error', 'Pitch Error', 'Roll Error']
fig4, axs4 = plt.subplots(1, 3, figsize=(18, 5))
for i in range(3):
    axs4[i].hist(error_eul[:, i], bins=20, color='r')
    axs4[i].set_xlabel('Error (degrees)', fontsize=12)
    axs4[i].set_ylabel('Frequency', fontsize=12)
    axs4[i].set_title(titles_error_eul[i], fontsize=14)
    axs4[i].tick_params(labelsize=24)

# ---------------------------
# 8. Convert Euler Angles to Quaternions and Plot Time Series (Corresponding to MATLAB Figure(5))
# ---------------------------
# Using scipy's Rotation module.
# MATLAB's eul2quat(deg2rad(...), 'ZYX') returns quaternions in the format [w, x, y, z],
# while scipy's as_quat() returns [x, y, z, w] by default. We reorder them to [w, x, y, z].
r_tagslam = R.from_euler('ZYX', eul_angles, degrees=True)
quats = r_tagslam.as_quat()
quats = np.column_stack((quats[:, 3], quats[:, 0], quats[:, 1], quats[:, 2]))

r_est = R.from_euler('ZYX', eul_est_angles, degrees=True)
quats_est = r_est.as_quat()
quats_est = np.column_stack((quats_est[:, 3], quats_est[:, 0], quats_est[:, 1], quats_est[:, 2]))

fig5, axs5 = plt.subplots(4, 1, figsize=(10, 15))
quat_titles = ['q_0 (Scalar Part)', 'q_1', 'q_2', 'q_3']
for i in range(4):
    axs5[i].plot(t, quats[:, i], color='r', linewidth=2, label='Tagslam')
    axs5[i].plot(t, quats_est[:, i], color='b', linestyle='--', linewidth=2, label='Estimate')
    axs5[i].set_title(quat_titles[i], fontsize=14, fontweight='bold')
    axs5[i].set_xlabel('Time (s)')
    axs5[i].set_ylabel(quat_titles[i])
    axs5[i].legend(fontsize=12, loc='best')
    axs5[i].grid(True)
    axs5[i].tick_params(labelsize=24)

# ---------------------------
# 9. Plot Histogram of Quaternion Errors (Corresponding to MATLAB Figure(6))
# ---------------------------
quat_diff = quats - quats_est
fig6_quat, axs6_quat = plt.subplots(1, 4, figsize=(20, 5))
error_titles_quat = ['q_0 Error', 'q_1 Error', 'q_2 Error', 'q_3 Error']
for i in range(4):
    axs6_quat[i].hist(quat_diff[:, i], bins=20, color='r')
    axs6_quat[i].set_xlabel('Error', fontsize=12)
    axs6_quat[i].set_ylabel('Frequency', fontsize=12)
    axs6_quat[i].set_title(error_titles_quat[i], fontsize=14)
    axs6_quat[i].tick_params(labelsize=24)

# ---------------------------
# 10. 3D Trajectory Plot of Estimated Positions (Corresponding to MATLAB Figure(7))
# ---------------------------
fig7 = plt.figure()
ax7 = fig7.add_subplot(111, projection='3d')
ax7.plot(XYZ_est[:, 0], XYZ_est[:, 1], XYZ_est[:, 2],
         'r-', linewidth=2, label='Estimated')
ax7.scatter(XYZ_est[:, 0], XYZ_est[:, 1], XYZ_est[:, 2],
            s=40, c='r', marker='o')
ax7.set_xlabel('X', fontsize=14)
ax7.set_ylabel('Y', fontsize=14)
ax7.set_zlabel('Z', fontsize=14)
ax7.legend(fontsize=14)
ax7.set_title('Path of Estimated Positions', fontsize=16)
ax7.grid(True)
ax7.tick_params(labelsize=24)

# ---------------------------
# Show all figures
# ---------------------------
plt.show()
