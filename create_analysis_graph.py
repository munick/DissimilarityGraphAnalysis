__author__ = 'nmew'
__version__ = 0.1
import sys
sys.path.insert(0,'/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages')
import os
import logging
import numpy as np
from graph_tool.all import *
from analyzegraph import *

def main():
    parser = arguments()
    # single dimension
    # params = parser.parse_args(['', 'output/test/{0}_graphs/', 'output/test/', '-pos', 'output/test/2_coordinates.npy'])
    # range of dimensions
    # params = parser.parse_args(['', 'output/test/{0}_graphs/', 'output/test/', '-pos', 'output/test/2_coordinates.npy', '-d', '6', '8'])
    # 'output/dali/{0}_graphs/ output/dali/ -pos output/dali/2_coordinates.npy -d 2 20'
    # dali_2_4 = 'output/dali/{0}_graphs/ output/dali/ -pos output/dali/2_coordinates.npy -d 2 4'.split()
    # dali_24 = 'adsf output/final/dali/{0}_graphs/ output/final/dali/ -pos output/final/dali/24_coordinates.npy -d 24'.split()
    # dali_23_24 = 'adsf output/final/dali/{0}_graphs/ output/final/dali/ -pos output/final/dali/24_coordinates.npy -d 23 24'.split()
    # params = parser.parse_args(dali_24)
    params = parser.parse_args(sys.argv)
    logging.basicConfig(stream=params.log)
    global log
    log = logging.getLogger('analyzegraph')
    log.setLevel(logging.DEBUG)

    dissimilarityInputFile = params.dissimilarityInputFile
    # dissimilarityInputFile = "data/test/dali500.dist"
    graphFilenamePattern = params.graphFilenamePattern
    graphInputDirectory = params.graphInputDirectory
    outputDirectory = params.outputDirectory
    positionFile = params.positions
    graphNames = params.graphNames

    minDim = params.dimensions[0]
    if len(params.dimensions) == 2:
        maxDim = params.dimensions[1]
    else:
        maxDim = minDim

    graph = generateGraph(graphInputDirectory, graphFilenamePattern, positionFile, outputDirectory, graphNames, minDim, maxDim)
    saveGraph(graph, minDim, maxDim, outputDirectory)

    # log.info("creating graph images")
    # imagedir = outputDirectory + "graphImages/"
    # if not os.path.exists(imagedir):
    #     os.makedirs(imagedir)
    # output graph images
    # calcAndDrawBetweeness(graph, minDim, maxDim, graphNames, outputDirectory)
    # for dim in range(minDim, maxDim+1):
    #     for gname in graphNames:
    #         propertyName = proxiGraphPMName(gname, dim)
    #         filter = graph.edge_properties[propertyName]
    #         # outputProxiGraphOverlaps(graph, ['gg', '2nn'], dim, outputDirectory + "rng_gg_{0}d_overlap.png".format(minDim))
    #         graph_draw(graph, pos=graph.vertex_properties['pos'], output_size=(800, 800), vertex_size=8,
    #                    edge_pen_width=filter, output=imagedir + "{1}_{0}d_overlap.png".format(dim, gname))


def arguments():
    # argparse requires python 2.7
    from argparse import ArgumentParser
    from argparse import FileType
    #  "Run MDS on input distance file and save to output file"
    parser = ArgumentParser(prog="Dissimilarity Graph Analysis - Graph MDS", version=__version__)
    parser.add_argument('pythonScript', metavar='pythonScript')
    parser.add_argument('graphInputDirectory',
                        type=str,
                        help='Path to directory containing graph edge files. If range of dimensions specified, use {0} '
                             'to denote string to replace with dimension.')
    parser.add_argument('outputDirectory',
                        type=str,
                        default='output/defaultDir/',
                        help='Path to output directory.')
    parser.add_argument('-g', '--graphNames',
                        type=str,
                        default=['gg', 'rng', '4nn', '3nn', '2nn', '1nn'],
                        nargs='?',
                        help='List graphs to load. They will be added in the same order. '
                             'These are used in the graph filename pattern.')
    parser.add_argument('-p', '--graphFilenamePattern',
                        type=str,
                        default='{1}_{0}d.csv',
                        help='{0} denotes dimension, {1} denotes graph name')
    parser.add_argument('-d', '--dimensions',
                        type=int,
                        default=[2],
                        nargs='*',
                        help='Dimensions to load. If one value, that will be the dim loaded, '
                             'if two they are treated as min to max.')
    parser.add_argument('-diss', '--dissimilarityInputFile',
                        type=str,
                        help='Path to dissimilarity file')
    parser.add_argument('-pos', '--positions',
                        type=str,
                        help='2d x y coordinates for each vertex.')
    parser.add_argument('-log', '--log',
                        type=FileType,
                        default=sys.stderr,
                        nargs=1,
                        help='Path to log errors, info, etc. [default: %(default)s]')
    return parser


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error: {0}".format(e))
        raise
        # return -1