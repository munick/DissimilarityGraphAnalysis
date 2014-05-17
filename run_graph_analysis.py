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
    # dali_2_9_all_graphs = "run_graph_analysis.py output/final/dali/graph_2-9d.xml.gz output/final/dali/ -d 2 9".split()
    dali_2_4_all_graphs = "run_graph_analysis.py output/dali/graph_2-4d.xml.gz output/dali/ -d 2 4".split()
    dali_3_3d = "run_graph_analysis.py output/dali/graph_2-4d.xml.gz output/dali/ -v3d output/final/dali/3_coordinates.npy".split()
    params = parser.parse_args(dali_2_4_all_graphs)
    # params = parser.parse_args(sys.argv)
    logging.basicConfig(stream=params.log)
    global log
    log = logging.getLogger('analyzegraph')
    log.setLevel(logging.DEBUG)

    graphFile = params.graphFile
    outputDirectory = params.outputDirectory
    graphNames = params.graphNames

    minDim = params.dimensions[0]
    if len(params.dimensions) == 2:
        maxDim = params.dimensions[1]
    else:
        maxDim = minDim

    # graph = load_graph(graphFile)

    ###### What's not changing?
    # Graph Intersections
    imagedir = outputDirectory + "graphIntersections/"
    if not os.path.exists(imagedir):
        os.makedirs(imagedir)
    # All dim intersction per graph
    # filenameTemplate = "{0}-{1}d_{2}_intersect.png"
    # draw_all_dim_intersections(graph, minDim, maxDim, graphNames, imagedir + filenameTemplate)
    # # All graph intersction per dim
    # filenameTemplate = "{0}d_" + "-".join(graphNames) + "_intersect.png"
    # draw_all_graph_intersections(graph, minDim, maxDim, graphNames, imagedir + filenameTemplate)
    # # # All graph intersection accross all dims
    # filenameTemplate = "{0}-{1}d_".format(minDim, maxDim) + "-".join(graphNames) + "_intersect.png"
    # draw_all_graph_intersections(graph, minDim, maxDim, graphNames, imagedir + filenameTemplate)
    #
    # # plot index centrality for all dimensions
    filenameTemplate = "{0}-{1}d_".format(minDim, maxDim) + "{0}_betweenness_plot.png"
    # betweenness = plot_centrality_across_dim_per_graph(graph, minDim, maxDim, graphNames, imagedir + filenameTemplate)

    if params.visualizeIn3d:
        graph3d(params.visualizeIn3d)
        # graph3d(params.visualizeIn3d, betweenness)


def graph3d(coordinates3d, graph=None, dimension=None, betweenness=None):
    from mayavi import mlab
    log.info("visualizing 3d")
    positions = np.load(coordinates3d)
    xyz = positions.T
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    mlab.figure(1, bgcolor=(0, 0, 0))
    if graph is not None and dimension is not None and betweenness is not None:
        s = betweenness[proxiGraphPMName(graph, dimension)]
        mlab.points3d(x, y, z, s, colormap="copper", scale_factor=.25)
    else:
        mlab.points3d(x, y, z, colormap="copper", scale_factor=.25)
    # todo: add line connections
    mlab.show()


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
    parser.add_argument('-v3d', '--visualizeIn3d',
                        type=str,
                        default='',
                        nargs='*',
                        help='3d xyz coordinates for each vertex. If set, will export certain graphs in 3d '
                             'format to visualize in Mayavi (http://docs.enthought.com/mayavi/mayavi/index.html).')
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