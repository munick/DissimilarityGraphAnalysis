__version__ = '0.1'

import sys
sys.path.insert(0, '/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages')
import os
import csv
import logging
import time
import numpy as np
import proximitygraph as pg


def main():
    parser = arguments()
    params = parser.parse_args(sys.argv)
    logging.basicConfig(stream=params.log)

    outputDirectory = params.outputDirectory
    # inputFileName = os.path.splitext(os.path.basename(params.coordinateInputFile))

    # input coordinate file
    with np.load(params.coordinateInputFile) as data:
        coordinates = data

    # dimensions
    dimensions = coordinates.shape[1]
    fSuffix = "_({0})".format(dimensions)

    # get distances
    distanceMatrix = pg.gen_distance_matrix(coordinates)

    # run each graph and store edges
    allEdges, edgeCount = pg.get_complete_graph(distanceMatrix)
    
    # gg
    start_time = time.time()
    print("gg start")
    ggEdges, ggEdgeCount = pg.get_gabriel_graph(allEdges, edgeCount, distanceMatrix)
    runtime = time.time() - start_time
    print("gg end {0}".format(runtime))
    ggRealEdgeCount = len(ggEdges)
    print("gg edge-count {0} v {1}".format(ggEdgeCount, ggRealEdgeCount))
    # store
    save_graph(ggEdges, 'gg', outputDirectory, fSuffix)

    # rng
    start_time = time.time()
    print("rng start")
    rngEdges, rngEdgeCount = pg.get_relative_neighbor_graph(ggEdges, ggEdgeCount, distanceMatrix)
    runtime = time.time() - start_time
    print("rng end {0}".format(runtime))
    print("rng edge-count {0}".format(rngEdgeCount))
    # store
    save_graph(rngEdges, 'rng', outputDirectory, fSuffix)

    # nn
    for k in range(1, 5):
        start_time = time.time()
        print("{0}nn start".format(k))
        nnkEdges, nnkEdgeCount = pg.get_nearest_k_neighbor_graph(distanceMatrix, k)
        runtime = time.time() - start_time
        print("{0}nn end {1}".format(k, runtime))
        print("{0}nn edge-count {1}".format(k, nnkEdgeCount))
        # store
        save_graph(nnkEdges, "{0}nn".format(k), outputDirectory, fSuffix)


def save_graph(edges, graphType, outputDirectory, filenameSuffix=''):
    # create dir if not already there
    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    with open(outputDirectory + graphType + filenameSuffix + '.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target"])
        for edge in edges:
            writer.writerow(list(edge))


def arguments():
    # argparse requires python 2.7
    from argparse import ArgumentParser
    from argparse import FileType
    #  "Run MDS on input distance file and save to output file"
    parser = ArgumentParser(prog="Dissimilarity Graph Analysis - Graph MDS", version=__version__)
    parser.add_argument('--version', action='version', version='%(prog)s v%(version)s')
    parser.add_argument('pythonScript', metavar='pythonScript')
    parser.add_argument('coordinateInputFile',
                        type=str,
                        help='Path to coordinate file')
    parser.add_argument('outputDirectory',
                        type=str,
                        default='output/defaultDir/',
                        help='Path to output directory.')
    parser.add_argument('-log', '--log',
                        type=FileType,
                        default=sys.stderr,
                        nargs='?',
                        help='Path to log errors, info, etc. [default: %(default)s]')


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error: {0}".format(e))
        raise
        # return -1