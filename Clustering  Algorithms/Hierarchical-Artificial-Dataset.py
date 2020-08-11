import time
import warnings
from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np
from sklearn import metrics
n_samples = 1500
random_state = 170

np.random.seed(0)
#make and plot circles 
noisy_circles_x,noisy_circles_y = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)


#make and plot moons
noisy_moons_x,noisy_moons_y = datasets.make_moons(n_samples=n_samples, noise=.05)


#make and plot blobs
blobs_x,blobs_y = datasets.make_blobs(n_samples=n_samples, random_state=8)


#make and plot no structure 
no_structure_x,no_structure_y = np.random.rand(n_samples, 2), None


# blobs with varied variances
varied_x,varied_y = datasets.make_blobs(n_samples=n_samples,cluster_std=[1.0, 2.5, 0.5],random_state=random_state)


# Anisotropicly distributed data
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples,random_state = random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)


#make and plot circles
plt.figure(figsize=(10,7))

plt.scatter(noisy_circles_x[:,0],noisy_circles_x[:,1], s=10, c='black',edgecolors = "black", label='Items')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Noisy Circle')

plt.legend()
plt.show()

xx_circles = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'single').fit(noisy_circles_x)

plt.figure(figsize=(10,7))

plt.scatter(noisy_circles_x[:,0],noisy_circles_x[:,1], s=10, c='black',edgecolors = "black", label='unassigned')
plt.scatter(noisy_circles_x[xx_circles.labels_ == 0,0],noisy_circles_x[xx_circles.labels_ == 0,1], s=10, c='red',edgecolors = "black", label='1st_cluster')
plt.scatter(noisy_circles_x[xx_circles.labels_ == 1,0],noisy_circles_x[xx_circles.labels_ == 1,1], s=10, c='green',edgecolors = "black", label='2nd_cluster')
plt.scatter(noisy_circles_x[xx_circles.labels_ == -1,0],noisy_circles_x[xx_circles.labels_ == -1,1], s=10, c='blue',edgecolors = "black", label='outliers')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Noisy Circle Cluster')

plt.legend()
plt.show()

#make and plot moons
plt.figure(figsize=(10,7))

plt.scatter(noisy_moons_x[:,0],noisy_moons_x[:,1], s=10, c='black',edgecolors = "black", label='Items')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Noisy Moons')

plt.legend()
plt.show()

xx_moons = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'single').fit(noisy_moons_x)

plt.figure(figsize=(10,7))

plt.scatter(noisy_moons_x[:,0],noisy_moons_x[:,1], s=10, c='blue',edgecolors = "black", label='unassigned')
plt.scatter(noisy_moons_x[xx_moons.labels_ == 0,0],noisy_moons_x[xx_moons.labels_ == 0,1], s=10, c='red',edgecolors = "black", label='1st_cluster')
plt.scatter(noisy_moons_x[xx_moons.labels_ == 1,0],noisy_moons_x[xx_moons.labels_ == 1,1], s=10, c='green',edgecolors = "black", label='2nd_cluster')
plt.scatter(noisy_moons_x[xx_moons.labels_ == -1,0],noisy_moons_x[xx_moons.labels_ == -1,1], s=10, c='black',edgecolors = "black", label='outliers')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Noisy Moons Cluster')

plt.legend()
plt.show()

#make and plot blobs
plt.figure(figsize=(10,7))

plt.scatter(blobs_x[:,0],blobs_x[:,1], s=10, c='black',edgecolors = "black", label='Items')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Blobs')

plt.legend()
plt.show()

xx_blobs = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'single').fit(blobs_x)

plt.figure(figsize=(10,7))

plt.scatter(blobs_x[:,0],blobs_x[:,1], s=10, c='blue',edgecolors = "black", label='unassigned')
plt.scatter(blobs_x[xx_blobs.labels_ == 0,0],blobs_x[xx_blobs.labels_ == 0,1], s=10, c='red',edgecolors = "black", label='1st_cluster')
plt.scatter(blobs_x[xx_blobs.labels_ == 1,0],blobs_x[xx_blobs.labels_ == 1,1], s=10, c='green',edgecolors = "black", label='2nd_cluster')
plt.scatter(blobs_x[xx_blobs.labels_ == 2,0],blobs_x[xx_blobs.labels_ == 2,1], s=10, c='yellow',edgecolors = "black", label='3rd_cluster')
plt.scatter(blobs_x[xx_blobs.labels_ == -1,0],blobs_x[xx_blobs.labels_ == -1,1], s=10, c='black',edgecolors = "black", label='outliers')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Blobs Cluster')

plt.legend()
plt.show()

#make and plot no structure

plt.figure(figsize=(10,7))

plt.scatter(no_structure_x[:,0],no_structure_x[:,1], s=10, c='black',edgecolors = "black", label='Items')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('No Structure')

plt.legend()
plt.show()

xx_structure = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward').fit(no_structure_x)

plt.figure(figsize=(10,7))

plt.scatter(no_structure_x[:,0],no_structure_x[:,1], s=10, c='blue',edgecolors = "black", label='unassigned')
plt.scatter(no_structure_x[xx_structure.labels_ == 0,0],no_structure_x[xx_structure.labels_ == 0,1], s=10, c='red',edgecolors = "black", label='1st_cluster')
plt.scatter(no_structure_x[xx_structure.labels_ == 1,0],no_structure_x[xx_structure.labels_ == 1,1], s=10, c='green',edgecolors = "black", label='2nd_cluster')
plt.scatter(no_structure_x[xx_structure.labels_ == 2,0],no_structure_x[xx_structure.labels_ == 2,1], s=10, c='yellow',edgecolors = "black", label='3rd_cluster')
plt.scatter(no_structure_x[xx_structure.labels_ == -1,0],no_structure_x[xx_structure.labels_ == -1,1], s=10, c='black',edgecolors = "black", label='outliers')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('No Structure Cluster')

plt.legend()
plt.show()

# blobs with varied variances

plt.figure(figsize=(10,7))

plt.scatter(varied_x[:,0],varied_x[:,1], s=10, c='black',edgecolors = "black", label='Items')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Varied')

plt.legend()
plt.show()

xx_varied = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward').fit(varied_x)

plt.figure(figsize=(10,7))

plt.scatter(varied_x[:,0],varied_x[:,1], s=10, c='black',edgecolors = "black", label='unassigned')
plt.scatter(varied_x[xx_varied.labels_ == 0,0],varied_x[xx_varied.labels_ == 0,1], s=10, c='red',edgecolors = "black", label='1st_cluster')
plt.scatter(varied_x[xx_varied.labels_ == 1,0],varied_x[xx_varied.labels_ == 1,1], s=10, c='green',edgecolors = "black", label='2nd_cluster')
plt.scatter(varied_x[xx_varied.labels_ == 2,0],varied_x[xx_varied.labels_ == 2,1], s=10, c='yellow',edgecolors = "black", label='3rd_cluster')
plt.scatter(varied_x[xx_varied.labels_ == -1,0],varied_x[xx_varied.labels_ == -1,1], s=10, c='black',edgecolors = "black", label='outliers')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Varied Cluster')

plt.legend()
plt.show()

# Anisotropicly distributed data

plt.figure(figsize=(10,7))

plt.scatter(X_aniso[:,0],X_aniso[:,1], s=10, c='black',edgecolors = "black", label='Items')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Aniso')

plt.legend()
plt.show()

xx_aniso = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward').fit(X_aniso)

plt.figure(figsize=(10,7))

plt.scatter(X_aniso[:,0],X_aniso[:,1], s=10, c='black',edgecolors = "black", label='unassigned')
plt.scatter(X_aniso[xx_aniso.labels_ == 0,0],X_aniso[xx_aniso.labels_ == 0,1], s=10, c='red',edgecolors = "black", label='1st_cluster')
plt.scatter(X_aniso[xx_aniso.labels_ == 1,0],X_aniso[xx_aniso.labels_ == 1,1], s=10, c='green',edgecolors = "black", label='2nd_cluster')
plt.scatter(X_aniso[xx_aniso.labels_ == 2,0],X_aniso[xx_aniso.labels_ == 2,1], s=10, c='yellow',edgecolors = "black", label='3rd_cluster')
plt.scatter(X_aniso[xx_aniso.labels_ == -1,0],X_aniso[xx_aniso.labels_ == -1,1], s=10, c='black',edgecolors = "black", label='outliers')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Aniso Cluster')

plt.legend()
plt.show()