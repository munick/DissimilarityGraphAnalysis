__author__ = 'nmew'
__version__ = 0.1
import sys
sys.path.insert(0,'/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages')
import os
import logging
import numpy as np
from graph_tool.all import *
from analyzegraph import *
# from gi.repository import Gtk, Gdk, GdkPixbuf, GObject

def main():
    parser = arguments()
    # single dimension
    # params = parser.parse_args(['', 'output/test/{0}_graphs/', 'output/test/', '-pos', 'output/test/2_coordinates.npy'])
    # range of dimensions
    # params = parser.parse_args(['', 'output/test/{0}_graphs/', 'output/test/', '-pos', 'output/test/2_coordinates.npy', '-d', '6', '8'])
    params = parser.parse_args(sys.argv)
    logging.basicConfig(stream=params.log)
    global log
    log = logging.getLogger('analyzegraph')
    log.setLevel(logging.DEBUG)

    graphFile = params.graphFile
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

    graph = load_graph(graphFile)



def arguments():
    # argparse requires python 2.7
    from argparse import ArgumentParser
    from argparse import FileType
    #  "Run MDS on input distance file and save to output file"
    parser = ArgumentParser(prog="Dissimilarity Graph Analysis - Graph MDS", version=__version__)
    parser.add_argument('pythonScript', metavar='pythonScript')
    parser.add_argument('graphFile',
                        type=str,
                        help='Path to previously created and saved graph file.')
    parser.add_argument('outputDirectory',
                        type=str,
                        default='output/defaultDir/',
                        help='Path to output directory.')
    parser.add_argument('-g', '--graphNames',
                        type=str,
                        default=['gg', 'rng', '4nn', '3nn', '2nn', '1nn'],
                        nargs='?',
                        help='List graph types saved in file.')
    parser.add_argument('-d', '--dimensions',
                        type=int,
                        default=[2-19],
                        nargs='*',
                        help='Dimensions loaded in file.')
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