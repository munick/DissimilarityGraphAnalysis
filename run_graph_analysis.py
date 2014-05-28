__author__ = 'nmew'
__version__ = 0.1
import sys

sys.path.insert(0, '/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages')
import os
import logging
import csv
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
    # dali_2_4_all_graphs = "run_graph_analysis.py output/dali/graph_2-4d.xml.gz output/dali/ -d 2 4".split()
    # dali_3_3d = "run_graph_analysis.py output/dali/graph_2-4d.xml.gz output/dali/ -v3d output/final/dali/3_coordinates.npy".split()
    # testBtwn = "run_graph_analysis.py output/javaMdsj/dali/graph_2-6d.xml.gz output/javaMdsj/dali/ -d 2 6 -btwn".split()
    testDeg = "run_graph_analysis.py output/javaMdsj/dali/graph_2-6d.xml.gz output/javaMdsj/dali/ -d 5 6 -deg".split()
    # testDeg = "run_graph_analysis.py output/javaMdsj/dali/graph_2-6d.xml.gz output/javaMdsj/dali/ -d 2 4 -btwn".split()
    testInt = "run_graph_analysis.py output/javaMdsj/dali/graph_2-6d.xml.gz output/javaMdsj/dali/ -d 4 6 -intersect".split()
    testnprops = "run_graph_analysis.py output/javaMdsj/dali/graph_2-6d.xml.gz output/javaMdsj/dali_analysis/ -d 5 6 -nprops".split()
    testAll = "run_graph_analysis.py output/javaMdsj/dali/graph_2-6d.xml.gz output/javaMdsj/dali_analysis/ -d 2 6 -nprops -intersect -btwn -deg -close".split()
    # params = parser.parse_args(testnprops)
    params = parser.parse_args(sys.argv)
    logging.basicConfig(stream=params.log)
    global log
    log = logging.getLogger('analyzegraph')
    log.setLevel(logging.DEBUG)

    # set min and max dimensions
    minDim = params.dimensions[0]
    if len(params.dimensions) == 2:
        maxDim = params.dimensions[1]
    else:
        maxDim = minDim

    # paths
    graphFile = params.graphFile
    outputDirectory = params.outputDirectory
    graphNames = params.graphNames

    # create output dir
    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    # only created if requested
    def getImageDir():
        imageDir = outputDirectory + "graphImages/"
        if not os.path.exists(imageDir):
            log.info("creating graph images dir")
            os.makedirs(imageDir)
        return imageDir

    # only created if requested
    def getPlotDir():
        new_dir = outputDirectory + "graphPlots/"
        if not os.path.exists(new_dir):
            log.info("creating graph plots dir")
            os.makedirs(new_dir)
        return new_dir

    # load the graph from input
    log.info("loading graph from file")
    graph = load_graph(graphFile)
    log.info("loading complete")
    graph.is_dirty = False      # flag for when we should save any changes, like new properties, to the graph

    def analyzeVertexPropterty(vertexProperty):
        log.info("analyzing " + vertexProperty)
        # output graph images
        drawGraph(graph, minDim, maxDim, graphNames, getImageDir(), vertexProperty)

        # create plot dir
        plot_dir = getPlotDir() + vertexProperty + '/'
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        # plot index centrality for all dimensions
        filenameTemplate = plot_dir + vertexProperty + "_{0}-{1}d_".format(minDim, maxDim) + "{0}_plot.png"
        minMaxAvg = plot_vertexproperty_across_dim_per_graph(vertexProperty, graph, minDim, maxDim, graphNames,
                                                             filenameTemplate)
        # write to csv
        for gType in graphNames:
            with open(plot_dir + vertexProperty + "_{0}-{1}d_{2}_".format(minDim, maxDim, gType) + '.csv', 'wb') as f:
                writer = csv.writer(f)
                writer.writerow(['graph_type', 'min_dim', 'max_dim', 'min_' + vertexProperty, 'max_' + vertexProperty,
                                 'avg_' + vertexProperty,
                                 'variation'])
                for i in range(graph.num_vertices()):
                    writer.writerow([gType, minDim, maxDim,
                                     minMaxAvg[gType]['min'][i],
                                     minMaxAvg[gType]['max'][i],
                                     minMaxAvg[gType]['avg'][i],
                                     minMaxAvg[gType]['max'][i] - minMaxAvg[gType]['min'][i]])

    if params.closeness:
        analyzeVertexPropterty('closeness')

    if params.betweenness:
        analyzeVertexPropterty('betweenness')

    if params.degree:
        analyzeVertexPropterty('degree')

    ###### What's changing/not changing?
    if params.intersection:
        log.info("analyzing intersection")
        imageDir = getImageDir()

        # All dim intersections per graph
        filenameTemplate = "{0}-{1}d_{2}_intersect.png"
        draw_all_dim_intersections(graph, minDim, maxDim, graphNames, imageDir + filenameTemplate)

        # # All graph intersctions per dim
        filenameTemplate = "{0}d_" + "-".join(graphNames) + "_intersect.png"
        draw_all_graph_intersections(graph, minDim, maxDim, ['rng', '1nn'], imageDir + filenameTemplate)

        # All graph intersection accross all dims
        filenameTemplate = "{0}-{1}d_".format(minDim, maxDim) + "-".join(['rng', '1nn']) + "_intersect.png"
        draw_all_graph_intersections(graph, minDim, maxDim, ['rng', '1nn'], imageDir + filenameTemplate)

    #### Network Properties and Averages
    if params.networkProperties:
        log.info("analyzing network properties")
        # define properties
        graphProps = ['size', 'density']
        vertexProps = ['degree', 'betweenness', 'closeness']
        # vertexProps = ['degree']
        props = graphProps + [prefix + prop for prop in vertexProps for prefix in ['min_', 'max_', 'avg_']]
        properties = {}
        # init properties dict
        for prop in props:
            properties[prop] = {}

        for dim in range(minDim, maxDim + 1):
            for gType in graphNames:
                pGraphName = proxiGraphPMName(gType, dim)
                # filter edges for dim & gType
                graph.set_edge_filter(None)
                efilter = graph.edge_properties[pGraphName]
                graph.set_edge_filter(efilter)
                # calc size and network density
                size = graph.num_edges()
                vcount = float(graph.num_vertices())
                potential_size = (vcount * (vcount - 1)) / 2
                density = size / potential_size
                properties['size'][pGraphName] = size
                properties['density'][pGraphName] = density
                # calc avg vertex Properties
                for vProperty in vertexProps:
                    propertyVertexValues = get_vertex_property(vProperty, graph, gType, dim)
                    averagePropertyValue = np.average(propertyVertexValues.a)
                    minimumPropertyValue = np.amin(propertyVertexValues.a)
                    maximumPropertyValue = np.amax(propertyVertexValues.a)
                    properties['min_' + vProperty][pGraphName] = minimumPropertyValue
                    properties['max_' + vProperty][pGraphName] = maximumPropertyValue
                    properties['avg_' + vProperty][pGraphName] = averagePropertyValue

        with open(outputDirectory + "{0}-{1}d_network_properties".format(minDim, maxDim) + '.csv', 'wb') as f:
            writer = csv.writer(f)
            writer.writerow(
                ['dimension', 'graph_type'] + graphProps + [prefix + prop for prop in vertexProps for prefix in
                                                            ['min_', 'max_', 'avg_']])
            for dim in range(minDim, maxDim + 1):
                for gType in graphNames:
                    pGraphName = proxiGraphPMName(gType, dim)
                    row = [dim, gType]
                    for prop in graphProps:
                        row.append(properties[prop][pGraphName])
                    for prop in vertexProps:
                        row.append(properties['min_' + prop][pGraphName])
                        row.append(properties['max_' + prop][pGraphName])
                        row.append(properties['avg_' + prop][pGraphName])
                    writer.writerow(row)


    # if properties were added, save them to disc so we don't have to calc them again
    if graph.is_dirty:
        log.info("saving graph properties")
        graph.save(graphFile)


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
                        default=[2 - 19],
                        nargs='*',
                        help='Dimensions loaded in file.')
    parser.add_argument('-close', '--closeness',
                        default=False,
                        action='store_true',
                        help='Calculate vertex degree')
    parser.add_argument('-deg', '--degree',
                        default=False,
                        action='store_true',
                        help='Calculate vertex degree')
    parser.add_argument('-btwn', '--betweenness',
                        default=False,
                        action='store_true',
                        help='Calculate vertex betweenness')
    parser.add_argument('-nprops', '--networkProperties',
                        default=False,
                        action='store_true',
                        help='Calculate various network Properties')
    parser.add_argument('-intersect', '--intersection',
                        default=False,
                        action='store_true',
                        help='Calculate gg&rng and gg&rng&nn1')
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