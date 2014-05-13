__author__ = 'nmew'
__version__ = 0.1
import sys
sys.path.insert(0,'/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages')
import os
import logging
import numpy as np
from graph_tool.all import *


logging.basicConfig(stream=sys.stderr)
global log
log = logging.getLogger('analyzegraph')
log.setLevel(logging.DEBUG)


def calcAndDrawBetweeness(graph, minDim, maxDim, graphNames, outputDirectory):
    log.info("creating graph images and betweenness")
    imagedir = outputDirectory + "graphImages/"
    if not os.path.exists(imagedir):
        os.makedirs(imagedir)
    log.info("\ncalculating betweenness of all graphs in all dimensions")
    for dim in range(minDim, maxDim+1):
        log.info("calculating betweenness of all {0}d graphs".format(dim))
        for gname in graphNames:
            log.info("\t{0} graph".format(gname))
            graph.set_edge_filter(None)
            propertyName = proxiGraphPMName(gname, dim)
            efilter = graph.edge_properties[propertyName]
            graph.set_edge_filter(efilter)
            bv, be = betweenness(graph)
            graph_draw(graph, pos=graph.vertex_properties['pos'], output_size=(800, 800), vertex_size=8,
                       vertex_fill_color=bv, output=imagedir + "{1}_{0}d.png".format(dim, gname))

def saveGraph(graph, minDim, maxDim, outputDirectory):
    graphFileName = "graph_{0}-{1}d.xml.gz".format(minDim, maxDim)
    graph.save(outputDirectory + graphFileName)
    log.info("graph saved as " + graphFileName)


def generateGraph(graphInputDirectory, graphFilenamePattern, positionFile, outputDirectory, graphNames, minDim, maxDim):
    log.info("loading positions")
    positions = np.load(positionFile)
    graph = gen_graph_with_2dpositions(positions)

    log.info("adding proximity graphs for dimensions {0}-{1}".format(minDim, maxDim))
    # for dimensions a -> z
    for dim in range(minDim, maxDim+1):
        log.info("start dim {0} ...".format(dim))
        graphDirectory = graphInputDirectory.format(dim)
        # for each proximity graphs
        for gname in graphNames:
            log.info("adding {0}".format(gname))
            edgeFileName = graphFilenamePattern.format(dim, gname)
            edgeFile = graphDirectory + edgeFileName
            edges = np.genfromtxt(edgeFile, skip_header=1, delimiter=',')
            # create edge property map
            addProxiGraphPM(graph, edges, gname, dim)

    log.info("graph created successfully")
    return graph


def outputProxiGraphOverlaps(graph, gnames, dim, filename):
    g = graph
    filters = []
    for gn in gnames:
        propertyName = proxiGraphPMName(gn, dim)
        filters.append(g.edge_properties[propertyName])

    # g.set_edge_filter(tree)
    # u = GraphView(g, efilt=lambda e: filters[0][e] + filters[1][e])
    pos = sfdp_layout(g, cooling_step=0.99)
    graph_draw(g, pos=g.vertex_properties['pos'], output_size=(800, 800), vertex_size=8,
               edge_pen_width=filters[1], output=filename)
               # edge_pen_width=lambda e: float(filters[0][e]) + float(filters[1][e]), output=filename)


# proximity graph property map name ie rng_3d, nn1_5d etc.
def proxiGraphPMName(gname, dim):
    return "{0}_{1}d".format(gname, dim)


def addProxiGraphPM(graph, edges, gname, dim):
    g = graph
    proxiGraphPM = g.new_edge_property("bool")  # boolean edge property
    # proxiGraphPM[g] = False                     # all edges should be false by default
    for edge in edges:                          # loop through edges and set to true
        # print(edge)
        g.add_edge(g.vertex(edge[0]), g.vertex(edge[1]))
        e = g.edge(edge[0], edge[1])
        proxiGraphPM[e] = True
    g.edge_properties[proxiGraphPMName(gname, dim)] = proxiGraphPM  # add to graph


def gen_graph_with_2dpositions(positions):
    log.info("creating graph from positions")
    # create graph
    g = Graph(directed=False)
    # create vertices
    g.add_vertex(positions.shape[0])
    # create pos propertymap
    posPM = g.new_vertex_property("vector<double>")  # vector vertex property map

    # for each row
    for i, xy in enumerate(positions):
        posPM[g.vertex(i)] = xy         # add x and y coordinates

    g.vertex_properties['pos'] = posPM  # add to graph
    return g


def gen_graph_from_dissimilarity(dissimilarityMatrix):
    # create graph
    g = Graph(directed=False)
    # create vertices
    g.add_vertex(dissimilarityMatrix.shape[0])


    # # todo: move this over to when we import edges from graphs to avoid creating all possible edges
    # # for each row
    # for i, row in enumerate(dissimilarityMatrix):
    # # and col
    #     for j in range(i+1, dissimilarityMatrix.shape[0]):
    #         # add edge with weight
    #         g.add_edge(g.vertex(i), g.vertex(j))
    return g


def genfromtxt(filename, skip_header=0, skip_column=0, dtype=np.float32, delimiter=','):
    with open(filename) as f:
        for it in range(skip_header+1):
            print("skipping header row {0}".format(it))
            f.readline()
        num_cols = len(f.readline().split(delimiter))
        print("num cols found: {0}\tdelimiter: {1}".format(num_cols, delimiter))
        # print("cols: {0}".format(num_cols))
    return np.genfromtxt(filename,
                         dtype=dtype, delimiter=delimiter, usecols=range(skip_column, num_cols),
                         skip_header=skip_header)
