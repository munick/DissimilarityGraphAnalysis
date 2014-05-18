import sys
sys.path.insert(0, '/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages')
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn import manifold
from sklearn.metrics import euclidean_distances

numberOfPoints = 1000
numberOfDims = 3

# Initialize a and b numpy arrays with coordinates and weights
# originalCoordinates = np.zeros((numberOfPoints, numberOfDims))
#
# # generate coordinates in straight diagonal line
# for i in range(numberOfPoints):
#     for j in range(numberOfDims):
#         originalCoordinates[i][j] = i
#
# # save to txt
# np.savetxt('originalCoordinates.txt', originalCoordinates)

# calc pairwise distance matrix
# distanceMatrix = squareform(pdist(originalCoordinates, 'euclidean'))

def main():
    distanceMatrix = genfromtxt('data/Pspace/dali.dist', skip_header=1, skip_column=1, to_column=800, to_header=800,
                                delimiter='\t')

    # now run mds
    mmds = manifold.MDS(metric=False, verbose=1, dissimilarity="precomputed", n_components=numberOfDims, max_iter=300,
                        n_jobs=-1, eps=0.001)
    newCoordinates = mmds.fit(distanceMatrix).embedding_

    # save new coordinates
    np.savetxt('daliTestCoordinatesNonMetric_sklearn.txt', newCoordinates)


def genfromtxt(filename, skip_header=0, skip_column=0,  to_header=0, to_column=0, dtype=np.float32, delimiter=','):
    with open(filename) as f:
        for it in range(skip_header):
            print("skipping header row {0}".format(it))
            f.readline()
        num_cols = len(f.readline().split(delimiter))

        if to_column <= 0:
            to_column = num_cols

        skip_footer = num_cols - to_header
        print("num cols to use: {0}\tdelimiter: {1}".format(to_column, delimiter))
        # print("cols: {0}".format(num_cols))
    return np.genfromtxt(filename,
                         dtype=dtype, delimiter=delimiter, usecols=range(skip_column, to_column),
                         skip_header=skip_header, skip_footer=skip_footer)


if __name__ == '__main__':
    main()