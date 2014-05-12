__author__ = 'nmew'
__version__ = '0.1'
"""
proximitygraph
=====================
Classes and methods defining graph types and converting similarity or distance matrices to sets of points and edges.
"""

import sys
sys.path.insert(0,'/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages')
import numpy as np
from scipy.spatial.distance import pdist, squareform
import itertools
import math
import multiprocessing
import time


_globalRelationalMatrix = np.zeros((1, 2))


def example():
    dim = 5
    points = 400
    print("gen {0} coordinates in {1} dimensions".format(points, dim))
    coordinates = gen_fake_coordinates(points, dim)

    print("gen distances")
    relationalMatrix = gen_distance_matrix(coordinates)

    print("gen all edges from {0} coordinates".format(relationalMatrix.shape[0]))
    allEdges, edgeCount = get_complete_graph(relationalMatrix)
    print("edgeCount: {0}".format(edgeCount))

    starttime = time.time()
    print("gg start")
    ggEdges = get_gabriel_graph(allEdges, edgeCount, relationalMatrix)
    runtime = time.time() - starttime
    print("gg end {0}".format(runtime))

    ggEdgeCount = len(ggEdges)
    print("gg edgecount {0}".format(ggEdgeCount))

    starttime = time.time()
    print("rng start")
    rngEdges = get_relative_neighbor_graph(ggEdges, ggEdgeCount, relationalMatrix)
    runtime = time.time() - starttime
    print("rng end {0}".format(runtime))

    rngEdgeCount = len(rngEdges)
    print("rng edgecount {0}".format(rngEdgeCount))

    starttime = time.time()
    print("5nn start")
    nn5Edges = get_nearest_k_neighbor_graph(relationalMatrix, 5)
    runtime = time.time() - starttime
    print("5nn end {0}".format(runtime))
    nn5EdgeCount = len(nn5Edges)
    print("5nn edgecount {0}".format(nn5EdgeCount))

    starttime = time.time()
    print("nn start")
    nnEdges = get_nearest_k_neighbor_graph(relationalMatrix, 1)
    runtime = time.time() - starttime
    print("nn end {0}".format(runtime))
    nnEdgeCount = len(nnEdges)
    print("5nn edgecount {0}".format(nnEdgeCount))


def gen_distance_matrix(coordinates, metric='euclidean'):
    print("calc distances of numpoints,dim={0}".format(coordinates.shape))
    coordinateRelationships = pdist(coordinates, metric)    # calc distances
    relationalMatrix = squareform(coordinateRelationships)  # make it a matrix
    np.fill_diagonal(relationalMatrix, np.NaN)              # replace diagonal with Nans
    return relationalMatrix


def completeSize(n):
    return (n * (n-1)) / 2


def gen_fake_coordinates(n_samples=50, dim=2):
    seed = np.random.RandomState(seed=3)
    X_true = seed.randint(0, 20, dim * n_samples).astype(np.float64)
    X_true = X_true.reshape((n_samples, dim))
    # Center the data
    X_true -= X_true.mean()
    return X_true


def get_complete_graph(relationMatrix):
    edges = itertools.combinations(range(relationMatrix.shape[0]), 2)
    edgeCount = completeSize(relationMatrix.shape[0])
    return edges, edgeCount


def get_nearest_k_neighbor_graph(relationMatrix, k):
    neighbors = []
    edges = set()
    # compute neighbors for each row
    if k > 1:
        neighbors = np.argsort(relationMatrix, axis=1)[:, 0:k]      # by row
    elif k == 1:
        neighbors = np.nanmin(relationMatrix, axis=1)[np.newaxis].T # by row

    for i, localneighbors in enumerate(neighbors):
        # print("{0} has {1} neighbors".format(i, len(localneighbors)))
        for neighbor in localneighbors:
            # if not np.isnan(neighbor):
            edges.add(frozenset((i, neighbor)))

    return edges, len(edges)


# gabriel graph
def get_gabriel_graph(inputEdges, edgeCount, relationMatrix):
    print("calculating gg in parallel ...")
    return parallel_proximity_graph(inputEdges, edgeCount, relationMatrix, gg_worker)


# relative neighbor graph
def get_relative_neighbor_graph(inputEdges, edgeCount, relationMatrix):
    print("calculating rng in parallel ...")
    return parallel_proximity_graph(inputEdges, edgeCount, relationMatrix, rng_worker)


def parallel_proximity_graph(inputEdges, edgeCount, relationMatrix, graphWorker, n_jobs=0):
    """
    Returns the relative neighborhood graph of the given relational matrix.

    :param relationMatrix: pairwise distance matrix
    :param getBestScore: returns highest similarity or smallest distance
    :param getWorstScore: returns lowest similarity or largest distance
    """
    print("Storing globals")
    global _globalRelationalMatrix
    _globalRelationalMatrix = relationMatrix                                # put relationMatrix in global mem
    print("Initializing parallel params")
    # inputEdges = list(inputEdges)
    edges = set()                                                           # set of edges found
    print("Initializing queue")
    queue = multiprocessing.Queue()                                         # results queue
    print("Determining CPU count")
    if n_jobs <= 0:
        n_jobs = max(1, multiprocessing.cpu_count() - n_jobs)
    numberOfProcesses = min(edgeCount, n_jobs)                              # number of processors to use
    print("CPU count: {0}".format(numberOfProcesses))
    print("Determining data per process")
    edgesPerProcess = edgeCount/numberOfProcesses                           # number of edges per processor
    inputEdgeSubsets = []                                                   # contains edge set for each process
    subsetIndex = -1
    print("Dividing data")
    # divide inputEdges into even subsets
    for i, edge in enumerate(inputEdges):
        if i % edgesPerProcess == 0:
            inputEdgeSubsets.append(set())
            subsetIndex += 1
        inputEdgeSubsets[subsetIndex].add(edge)

    print("Running Processes: {0}".format(numberOfProcesses))
    # run processes
    for i in range(numberOfProcesses):
        p = multiprocessing.Process(target=graphWorker, args=(inputEdgeSubsets[i], queue))
        p.start()

    # collect results
    for i in range(numberOfProcesses):
        edges = edges.union(queue.get())
        print("{0} process complete".format(i))

    return edges, len(edges)


# Relative Neighbor graph
def rng_worker(inputEdges, queue):
    """
    Work done by each processor
    :param inputEdges: set of edges (p, q)
    :param queue: shared Queue to place results
    """
    edges = set()
    for p, q in inputEdges:
        relationPQ = _globalRelationalMatrix[p, q]
        row = _globalRelationalMatrix[p]
        # maxJRow = getBestScore(relationMatrix[q])
        # non-numerical distances/similarities will not be counted as edges
        if np.isnan(relationPQ):
            isEdge = False
        # if there is a numeric value
        else:
            isEdge = True   # assume edge until proven wrong
            # loop through all columns in the ith row
            # relationPR is weight of edge p,r ***************************************************** (N^3)/2
            for r, relationPR in enumerate(row):
                # skip rows p and q and any points for which there is no distance value
                if p != r != q and (not np.isnan(relationPR)) and (not np.isnan(_globalRelationalMatrix[q, r])):
                    # for triangle prq, if pq is the longest distance, then p and q are not neighbors
                    lengths = [relationPR, _globalRelationalMatrix[q, r]]
                    if lengths[np.nanargmax(lengths)] < relationPQ:
                        isEdge = False      # not an edge!
                        break               # break to next q
        # if p and q are neighbors
        if isEdge:
            edges.add(frozenset((p, q)))    # add (p,q) tuple to edges set
    queue.put(edges)


# Gabriel graph
def gg_worker(inputEdges, queue):
    """
    Returns the gabriel graph of the given relational matrix.

    :param inputEdges: array of sets of 2 or tuples containing the vertex indices of an edge
    :param queue: multiprocessing queue to put results in
    """
    # edgesWithWeights = set()
    edges = set()
    # loop through rows of distance/similarity matrix ************************************************* N
    for p, q in inputEdges:
        relationPQ = _globalRelationalMatrix[p, q]
        row = _globalRelationalMatrix[p]
        # non-numerical distances/similarities will not be counted as edges
        if np.isnan(relationPQ):
            isEdge = False
        # if there is a numeric value
        else:
            isEdge = True # assume edge until proven wrong
            # loop through all columns in the ith row
            # relationPR is weight of edge p,r ***************************************************** (N^3)/2
            for r, relationPR in enumerate(row):
                # skip rows p and q and any where there are no distance values
                if p != r != q and (not np.isnan(relationPR)) and (not np.isnan(_globalRelationalMatrix[q, r])):
                    # if angle prq is > pi/2, then it's not an edge
                    # another calculation: if d(p,q) > sqrt of (d(p,r)^2 + d(r,q)^2)) then not an edge
                    lengths = math.sqrt(relationPR**2 + _globalRelationalMatrix[q, r]**2)
                    if relationPQ > lengths:
                        isEdge = False  # not an edge!
                        break           # break to next q
        # if p and q are neighbors
        if isEdge:
            edges.add(frozenset((p, q)))                            # add (p,q) tuple to edges set
            # edgesWithWeights.add((frozenset((p, q)), relationPQ))   # add ((p,q), weight) to weighted edges set

    queue.put(edges)


#
# # Gabriel graph single thread
# def getGabrielNeighborGraph(inputEdges, relationMatrix, getBestScore, getWorstScore):
#     """
#     Returns the gabriel graph of the given relational matrix.
#
#     :param relationMatrix: pairwise distance matrix
#     :param getBestScore: returns highest similarity or smallest distance
#     :param getWorstScore: returns lowest similarity or largest distance
#     """
#     edgesWithWeights = set()
#     edges = set()
#     # loop through rows of distance/similarity matrix ************************************************* N
#     for p, q in inputEdges:
#         relationPQ = relationMatrix[p, q]
#         row = relationMatrix[p]
#         # non-numerical distances/similarities will not be counted as edges
#         if np.isnan(relationPQ):
#             isEdge = False
#         # if there is a numeric value
#         else:
#             isEdge = True # assume edge until proven wrong
#             # loop through all columns in the ith row
#             # relationPR is weight of edge p,r ***************************************************** (N^3)/2
#             for r, relationPR in enumerate(row):
#                 # skip rows p and q and any where there are no distance values
#                 if p != r != q and (not np.isnan(relationPR)) and (not np.isnan(relationMatrix[q, r])):
#                     # if angle prq is > pi/2, then it's not an edge
#                     # another calculation: if d(p,q) > sqrt of (d(p,r)^2 + d(r,q)^2)) then not an edge
#                     lengths = math.sqrt(relationPR**2 + relationMatrix[q, r]**2)
#                     if relationPQ > lengths:
#                         isEdge = False  # not an edge!
#                         break           # break to next q
#         # if p and q are neighbors
#         if isEdge:
#             edges.add(frozenset((p, q)))                            # add (p,q) tuple to edges set
#             edgesWithWeights.add((frozenset((p, q)), relationPQ))   # add ((p,q), weight) to weighted edges set
#
#     return getPointGroupMapping(edges), edges

#
# # relative neighbor graph
# def getRelativeNeighborGraph(inputEdges, relationMatrix, getBestScore, getWorstScore):
#     """
#     Returns the relative neighborhood graph of the given relational matrix.
#
#     :param relationMatrix: pairwise distance matrix
#     :param getBestScore: returns highest similarity or smallest distance
#     :param getWorstScore: returns lowest similarity or largest distance
#     """
#     edges = set()
#     # loop through rows of distance/similarity matrix ************************************************* N
#     for p, q in inputEdges:
#         relationPQ = relationMatrix[p, q]
#         row = relationMatrix[p]
#         # maxJRow = getBestScore(relationMatrix[q])
#         # non-numerical distances/similarities will not be counted as edges
#         if np.isnan(relationPQ):
#             isEdge = False
#         # if there is a numeric value
#         else:
#             isEdge = True   # assume edge until proven wrong
#             # loop through all columns in the ith row
#             # relationPR is weight of edge p,r ***************************************************** (N^3)/2
#             for r, relationPR in enumerate(row):
#                 # skip rows p and q and any points for which there is no distance value
#                 if p != r != q and (not np.isnan(relationPR)) and (not np.isnan(relationMatrix[q, r])):
#                     # for triangle prq, if pq is the longest distance, then p and q are not neighbors
#                     lengths = [relationPR, relationMatrix[q, r]]
#                     if lengths[getWorstScore(lengths)] < relationPQ:
#                         isEdge = False  # not an edge!
#                         break           # break to next q
#         # if p and q are neighbors
#         if isEdge:
#             edges.add(frozenset((p, q)))                            # add (p,q) tuple to edges set
#
#     return getPointGroupMapping(edges), edges

# # nearest neighbor graph
# def getNearestNeighborGraph(relationMatrix, getBestScore):
#     edges = set()
#     for i, relationArray in enumerate(relationMatrix):
#         try:
#             nn = getBestScore(relationArray)
#             if not np.isnan(nn):
#                 edges.add(frozenset((i, nn)))
#         except ValueError:  # numpy 1.8 raises exception instead of warn & null value!
#             continue
#
#     return edges
#


# Identify Clusters/Neighborhoods in edges
def getPointGroupMapping(edges):
    pointGroupMapping = dict()
    pointGroups = dict()
    groupIndex = 0
    # loop through all edges
    for edge in edges:
        matchingGroupIndexes = []
        # find groups they belong to
        for index, group in pointGroups.iteritems():
            if not edge.isdisjoint(group):
                # they have points in common!
                matchingGroupIndexes.append(index)

        # if they don't belong to an existing group
        if len(matchingGroupIndexes) == 0:
            # add edges to new group
            pointGroups[groupIndex] = set(edge)
            groupIndex += 1

        # if they belong to one or more groups
        if len(matchingGroupIndexes) > 0:
            # take first index out of matchingGroupIndexes
            indexToKeep = matchingGroupIndexes.pop(0)
            # if matchingGroupIndexes has any more indexes
            for indexToRemove in matchingGroupIndexes:
                # merge them into indexToKeep and remove them from pointGroups
                pointGroups[indexToKeep].update(pointGroups.pop(indexToRemove))
            # finally, add edges to indexToKeep pointGroup
            pointGroups[indexToKeep].update(edge)

    # create point -> groupId mapping
    for groupId, points in enumerate(pointGroups.values()):
        for point in points:
            pointGroupMapping[point] = groupId

    return pointGroupMapping



if __name__ == "__main__":
    try:
        example()
    except Exception as e:
        print("Error: {0}".format(e))
        raise
        # return -1