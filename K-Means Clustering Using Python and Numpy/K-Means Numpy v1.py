import numpy as np
import matplotlib.pyplot as plt


def distance(center, point):
    return np.linalg.norm(center - point)


points = np.array([[-12, 12], [11, 3], [16, 9], [1, 14], [9, 0], [-2, -6], [0, -23], [-7, 5], [-10, -5], [-22, -8]])

# points = np.array([[0, 0], [0, 1], [1, 1], [1, 0], [9, 0], [10, 0], [9, 1], [10, 1], [11, 11]])

center_1 = points[0]
center_2 = points[8]

cluster_1 = np.empty((0, 2), int)
cluster_2 = np.empty((0, 2), int)

print("\n\nInitial Centroids:", center_1, center_2)

while True:
    for i in points:
        distance_center_1 = distance(center_1, i)
        distance_center_2 = distance(center_2, i)
        if distance_center_1 < distance_center_2:
            cluster_1 = np.append(cluster_1, [i], axis=0)
        else:
            cluster_2 = np.append(cluster_2, [i], axis=0)

    new_c1 = np.mean(cluster_1, axis=0)
    new_c2 = np.mean(cluster_2, axis=0)

    if np.array_equal(new_c1, center_1) and np.array_equal(new_c2, center_2):
        print("\n\nClustering completed successfully",
              "\nCentroids:", center_1, center_2,
              "\n\nCluster_1:\n", cluster_1, "\n\nCluster_2:\n", cluster_2)

        plt.plot(center_1[0], center_1[1], color="red", marker=(5, 1), markersize=12)
        plt.annotate("Center_1", xy=(center_1[0], center_1[1]))
        plt.scatter(cluster_1[:, 0], cluster_1[:, 1], label="cluster_1", c="red", marker="o")

        plt.plot(center_2[0], center_2[1], color="blue", marker=(5, 1), markersize=12)
        plt.annotate("Center_2", xy=(center_2[0], center_2[1]))
        plt.scatter(cluster_2[:, 0], cluster_2[:, 1], label="cluster_2", c="blue", marker="o")
        plt.legend(loc='upper right')
        plt.show()
        break
    else:
        center_1 = new_c1
        center_2 = new_c2
        print("\n\nCentroids have changed:", center_1, center_2,
              "\n\nCluster_1:\n", cluster_1, "\n\nCluster_2:\n", cluster_2, "\n")

    cluster_1 = np.empty((0, 2), int)
    cluster_2 = np.empty((0, 2), int)
