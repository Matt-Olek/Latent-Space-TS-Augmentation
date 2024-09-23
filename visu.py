import matplotlib.pyplot as plt
import numpy as np

mu1 = [1, 1, 1]
mu2 = [2, 2, 2]
mu3 = [3, 3, 1]

sigma1 = [1, 1, 1]
sigma2 = [1, 1, 1]
sigma3 = [1, 1, 1]

alpha = 0.6

sigma1 = alpha * alpha
sigma2 = alpha * alpha
sigma3 = alpha * alpha

# Create a figure and a 3D axis with 3 clusters of 100 points
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
data1 = np.random.normal(mu1, sigma1, (100, 3))
data2 = np.random.normal(mu2, sigma2, (100, 3))
data3 = np.random.normal(mu3, sigma3, (100, 3))

# Plot the data
ax.scatter(data1[:, 0], data1[:, 1], data1[:, 2], c="r", marker="o")
ax.scatter(data2[:, 0], data2[:, 1], data2[:, 2], c="b", marker="o")
ax.scatter(data3[:, 0], data3[:, 1], data3[:, 2], c="g", marker="o")

# D not show any grid
ax.grid(False)
ax.axis("off")

# transparent background
fig.patch.set_alpha(0)

fig.patch.set_facecolor("none")

fig.savefig("3dplot.png", transparent=True)
# Show the plot

plt.show()
