import matplotlib.pyplot as plt

from Analysis.Cluster import Cluster

circle = Cluster.make_blobs(1500,None)
print(circle)

#plt.scatter(circle[:, 0], circle[:, 1], marker='o', c=Y1,
#            s=25, edgecolor='k')
#plt.show()