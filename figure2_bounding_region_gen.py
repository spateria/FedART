import numpy as np
import matplotlib.pyplot as plt

def fuzzy_and(X, W):
    ret = []
    for i in range(len(X)):
        ret.append(min(X[i], W[i]))
    return np.array(ret)
    
def sum_norm(V):
    ret = 0
    for v in V:
        ret += abs(v)
    return ret

# Define the center point and range
center = np.array([0.5, 0.5]) #example category weight vector
rho = 0.8
gamma = 1.0
alpha = 0.1

range_val = 0.4

# Generate grid cross-points around the center
num_points = 10
x_coords = np.linspace(center[0] - range_val, center[0] + range_val, num_points)
y_coords = np.linspace(center[1] - range_val, center[1] + range_val, num_points)

# Create a meshgrid from x and y coordinates
X, Y = np.meshgrid(x_coords, y_coords)
points_array = np.column_stack((X.ravel(), Y.ravel()))

colr = []
for i in range(len(points_array)):
    p = points_array[i]
    numer = sum_norm(fuzzy_and(p, center))
    denom = alpha + sum_norm(center)
    colr.append(gamma * (numer/denom))

# Plot the generated points
plt.figure(figsize=(8, 6))
plt.scatter(points_array[:, 0], points_array[:, 1], c=colr)
plt.scatter(center[0], center[1], color='red', marker='x', label='Center Point (0.2, 0.4)')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()



colr2 = []
for i in range(len(points_array)):
    p = points_array[i]
    numer = sum_norm(fuzzy_and(p, center))
    denom = sum_norm(p)
    colr2.append(int(numer/denom >= rho))

# Plot the generated points
plt.figure(figsize=(8, 6))
plt.scatter(points_array[:, 0], points_array[:, 1], c=colr2)
plt.scatter(center[0], center[1], color='red', marker='x', label='Center Point (0.2, 0.4)')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()


chosen_p = []
colr3 = []
for j in range(len(colr2)):
    if colr2[j] !=0:
        chosen_p.append(points_array[j])
        colr3.append(colr[j])
chosen_p = np.array(chosen_p)


# Plot the generated points
plt.figure(figsize=(8, 6))
plt.scatter(chosen_p[:, 0], chosen_p[:, 1], c=colr3)
plt.scatter(center[0], center[1], color='red', marker='x', label='Center Point (0.2, 0.4)')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()

