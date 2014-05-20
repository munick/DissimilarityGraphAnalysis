__author__ = 'nmew'
__version__ = 0.1
import sys

sys.path.insert(0, '/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages')
import os
import logging
import numpy as np
from graph_tool.all import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(stream=sys.stderr)
global log
log = logging.getLogger('analyzegraph')
log.setLevel(logging.DEBUG)

completeGraphName = {
    'gg': 'Gabriel Graph',
    'rn': 'Relative Neighbor Graph',
    'rng': 'Relative Neighbor Graph',
    '4nn': '4-Nearest Neighbors Graph',
    '3nn': '3-Nearest Neighbors Graph',
    '2nn': '2-Nearest Neighbors Graph',
    '1nn': 'Nearest Neighbor Graph'
}


def draw_all_dim_intersections(graph, minDim, maxDim, graphNames, filenameTemplate):
    log.info("draw_all_dim_intersections")
    for gname in graphNames:
        graph.set_edge_filter(None)
        u = GraphView(graph)
        propertyName = proxiGraphPMName(gname, minDim)
        filter = graph.edge_properties[propertyName].a.base
        # filter each successive dim
        for dim in range(minDim + 1, maxDim + 1):
            propertyName = proxiGraphPMName(gname, dim)
            dfilter = graph.edge_properties[propertyName].a.base
            filter = np.logical_and(filter, dfilter)
        u = GraphView(u, efilt=filter)
        graph_draw(u, pos=graph.vertex_properties['pos'], output_size=(800, 800), vertex_size=8,
                   output=filenameTemplate.format(minDim, maxDim, gname))


def draw_all_graph_intersections(graph, minDim, maxDim, graphNames, filenameTemplate):
    log.info("draw_all_graph_intersections")
    # todo: separate the filter calcuation to separate method
    for dim in range(minDim, maxDim + 1):
        graph.set_edge_filter(None)
        u = GraphView(graph)
        propertyName = proxiGraphPMName(graphNames[0], dim)
        efilter = graph.edge_properties[propertyName].a.base
        # filter each successive dim
        for gname in graphNames:
            propertyName = proxiGraphPMName(gname, dim)
            dfilter = graph.edge_properties[propertyName].a.base
            efilter = np.logical_and(efilter, dfilter)
        u = GraphView(u, efilt=efilter)
        graph_draw(u, pos=graph.vertex_properties['pos'], output_size=(800, 800), vertex_size=8,
                   output=filenameTemplate.format(dim))


def draw_all_graph_and_dim_intersections(graph, minDim, maxDim, graphNames, filenameTemplate):
    log.info("draw_all_graph_and_dim_intersections")
    # todo: separate the filter calcuation to separate method
    graph.set_edge_filter(None)
    u = GraphView(graph)
    propertyName = proxiGraphPMName(graphNames[0], minDim)
    efilter = graph.edge_properties[propertyName].a.base
    for dim in range(minDim, maxDim + 1):
        # filter each successive dim
        for gname in graphNames:
            propertyName = proxiGraphPMName(gname, dim)
            dfilter = graph.edge_properties[propertyName].a.base
            efilter = np.logical_and(efilter, dfilter)
    u = GraphView(u, efilt=efilter)
    graph_draw(u, pos=graph.vertex_properties['pos'], output_size=(800, 800), vertex_size=8,
               output=filenameTemplate)


def plot_vertexproperty_across_dim_per_graph(vertexProperty, graph, minDim, maxDim, graphNames, filenameTemplate):
    log.info("plot_" + vertexProperty + "_across_dim_per_graph")
    graph.set_edge_filter(None)
    vertexCount = graph.num_vertices()
    x = range(vertexCount)
    propertyMinMaxAvg = {}
    # Save a nice dark grey as a variable
    almost_black = '#262626'
    ball_color = '#3182bd'
    bar_color = '#08306b'

    for gname in graphNames:
        betweenMin = np.empty(vertexCount)
        betweenMin.fill(np.nan)
        betweenMax = np.empty(vertexCount)
        betweenMax.fill(np.nan)
        betweenAvg = np.zeros(vertexCount)
        graph.set_edge_filter(None)
        # get betweenness each dim
        for dim in range(minDim, maxDim + 1):
            propertyName = proxiGraphPMName(gname, dim)
            efilter = graph.edge_properties[propertyName]
            graph.set_edge_filter(None)
            graph.set_edge_filter(efilter)
            log.info("get " + vertexProperty + " of " + propertyName)
            vp = get_vertex_property(vertexProperty, graph, gname, dim)
            betweenMin = np.fmin(vp.a.base, betweenMin)
            betweenMax = np.fmax(vp.a.base, betweenMax)
            betweenAvg = np.add(vp.a.base, betweenAvg)

        betweenAvg = betweenAvg / (1 + maxDim - minDim)
        propertyMinMaxAvg[gname] = {'min': betweenMin, 'max': betweenMax, 'avg': betweenAvg}
        log.info("creating figure")
        fig = plt.figure(figsize=(32, 6))
        log.info("plotting figure")

        betweenMaxErrorBar = betweenMax - betweenAvg
        betweenMinErrorBar = betweenAvg - betweenMin

        # replace large values
        betweenMaxErrorBar[betweenMaxErrorBar > 1000] = 1000
        betweenAvg[betweenAvg > 1000] = 1000
        # replace small values
        betweenMinErrorBar[betweenMin < 0] = 0
        betweenAvg[betweenAvg < -1000] = -1000

        plt.errorbar(x, betweenAvg, yerr=[betweenMinErrorBar, betweenMaxErrorBar], fmt='o', elinewidth=0.25,
                     linewidth=0.15, alpha=0.75, ecolor=bar_color, color=ball_color)
        # plt.xticks(x, x)
        ax = fig.axes[0]
        ax.locator_params(axis='x', nbins=74)
        maxAvgPlusPadding = np.max(betweenAvg) * 1.25
        twiceAvg = np.average(betweenAvg) * 2
        ytopIfMaxTooHigh = np.maximum(twiceAvg, maxAvgPlusPadding)
        maxError = np.max(betweenMax)
        # is it too high?
        if maxError > 1.25 * ytopIfMaxTooHigh:
            ytop = ytopIfMaxTooHigh
        else:
            ytop = maxError

        plt.ylim(-0.005, ytop)
        plt.xlim(xmin=-10)

        for spine in ['top', 'right', 'bottom', 'left']:
            ax.spines[spine].set_linewidth(0.5)
            ax.spines[spine].set_color(almost_black)

        # Change the labels to the off-black
        ax.xaxis.label.set_color(almost_black)
        ax.yaxis.label.set_color(almost_black)

        # Tweak spacing to prevent clipping of tick-labels
        # plt.tight_layout()
        # set title
        ax.set_title(vertexProperty.capitalize() + " variation of Vertices in " + completeGraphName[
            gname] + "s generated for " + str(minDim) + "d through " + str(maxDim) + "d MDS Embeddings", fontsize=24)

        log.info("saving figure to {0}".format(filenameTemplate.format(gname)))
        plt.savefig(filenameTemplate.format(gname), bbox_inches='tight')
    return propertyMinMaxAvg


# def plot_centrality_across_dim_per_graph(graph, minDim, maxDim, graphNames, filenameTemplate):
#     log.info("plot_centrality_across_dim_per_graph")
#     graph.set_edge_filter(None)
#     vertexCount = graph.num_vertices()
#     betweenness = {}    # same as property name
#     x = range(vertexCount)
#     for gname in graphNames:
#         betweenMin = np.empty(vertexCount)
#         betweenMax = np.empty(vertexCount)
#         betweenAvg = np.zeros(vertexCount)
#         graph.set_edge_filter(None)
#         # get betweenness each dim
#         for dim in range(minDim, maxDim+1):
#             propertyName = proxiGraphPMName(gname, dim)
#             efilter = graph.edge_properties[propertyName]
#             graph.set_edge_filter(None)
#             graph.set_edge_filter(efilter)
#             log.info("calc betweenness of " + propertyName)
#             bv = get_betweenness(graph, gname, dim)
#             betweenness[propertyName] = bv
#             betweenMin = np.fmin(bv.a.base, betweenMin)
#             betweenMax = np.fmax(bv.a.base, betweenMax)
#             betweenAvg = np.add(bv.a.base, betweenAvg)
#
#         betweenAvg = betweenAvg / (maxDim-minDim+1)
#         log.info("creating figure")
#         plt.figure()
#         log.info("plotting figure")
#         plt.errorbar(x, betweenAvg, yerr=[betweenMin, betweenMax], fmt='--o')
#         # plot(x, betweenAvg/maxDim-minDim+1)
#         # You can specify a rotation for the tick labels in degrees or with keywords.
#         plt.xticks(x, x)
#         # Pad margins so that markers don't get clipped by the axes
#         plt.margins(0.2)
#         # Tweak spacing to prevent clipping of tick-labels
#         plt.subplots_adjust(bottom=0.15)
#         log.info("saving figure to {0}".format(filenameTemplate.format(gname)))
#         plt.savefig(filenameTemplate.format(gname))
#         return betweenness


def drawGraph(graph, minDim, maxDim, graphNames, outputDirectory, vertexProperty):
    log.info("creating graph images with " + vertexProperty)
    imagedir = outputDirectory + vertexProperty + "/"
    if not os.path.exists(imagedir):
        os.makedirs(imagedir)
    log.info("\ndrawing " + vertexProperty + " of all graphs in all dimensions")
    for dim in range(minDim, maxDim + 1):
        for gname in graphNames:
            log.info("\t{0} graph".format(gname))
            graph.set_edge_filter(None)
            propertyName = proxiGraphPMName(gname, dim)
            efilter = graph.edge_properties[propertyName]
            graph.set_edge_filter(efilter)
            vp = get_vertex_property(vertexProperty, graph, gname, dim)
            graph_draw(graph, pos=graph.vertex_properties['pos'], output_size=(800, 800), vertex_size=8,
                       vertex_fill_color=vp, output=imagedir + "{1}_{0}d_{2}.png".format(dim, gname, vertexProperty))


# def drawGraphWithBetweenness(graph, minDim, maxDim, graphNames, outputDirectory, vertexColor):
#     log.info("creating graph images with betweenness")
#     imagedir = outputDirectory + "graphImages/"
#     if not os.path.exists(imagedir):
#         os.makedirs(imagedir)
#     log.info("\ndrawing betweenness of all graphs in all dimensions")
#     for dim in range(minDim, maxDim+1):
#         for gname in graphNames:
#             log.info("\t{0} graph".format(gname))
#             graph.set_edge_filter(None)
#             propertyName = proxiGraphPMName(gname, dim)
#             efilter = graph.edge_properties[propertyName]
#             graph.set_edge_filter(efilter)
#             bv, be = get_betweenness(graph, gname, dim)
#             graph_draw(graph, pos=graph.vertex_properties['pos'], output_size=(800, 800), vertex_size=8,
#                        vertex_fill_color=bv, output=imagedir + "{1}_{0}d.png".format(dim, gname))


def get_vertex_property(vertexProperty, graph, graphName, dimension):
    """
    Get any vertex property that we calculate and store in the graph like betweenness and degree
    :param vertexProperty: name of vertex property getter that matches get_[vertexProperty] ie get_betweenness
    :param graph: graph in question
    :param graphName: graph type/name in question ie gg, rng, etc.
    :param dimension: underlying dimension
    :return:
    """
    propertyGetter = globals()['get_' + vertexProperty]
    properties = propertyGetter(graph, graphName, dimension)
    if isinstance(properties, tuple):
        vprop = properties[0]
    else:
        vprop = properties
    return vprop


def get_betweenness(graph, graphName, dimension):
    """

    :rtype :class:`~graph_tool.PropertyMap`
    """
    betweennessSuffix = "_betweenness"
    propertyName = proxiGraphPMName(graphName, dimension) + betweennessSuffix
    propertyMap = graph.vertex_properties.get(propertyName)
    if propertyMap is None:
        # calc betweenness
        log.info("calculating betweenness of {0}d {1}".format(dimension, graphName))
        propertyMap = graph.new_vertex_property("double")
        bv, be = betweenness(graph, vprop=propertyMap)
        graph.edge_properties[propertyName] = be
        graph.vertex_properties[propertyName] = propertyMap
        graph.is_dirty = True
    return graph.vertex_properties[propertyName], graph.edge_properties[propertyName]


def get_degree(graph, graphName, dimension):
    """

    :rtype :class:`~graph_tool.PropertyMap`
    """
    degreeSuffix = "_degree"
    propertyName = proxiGraphPMName(graphName, dimension) + degreeSuffix
    propertyMap = graph.vertex_properties.get(propertyName)
    if propertyMap is None:
        # calc betweenness
        log.info("calculating degree of {0}d {1}".format(dimension, graphName))
        dv = graph.degree_property_map("total")
        graph.vertex_properties[propertyName] = dv
        graph.is_dirty = True
    return graph.vertex_properties[propertyName]


def get_closeness(graph, graphName, dimension):
    suffix = "_closeness"
    propertyName = proxiGraphPMName(graphName, dimension) + suffix
    propertyMap = graph.vertex_properties.get(propertyName)
    if propertyMap is None:
        # calc closeness
        log.info("calculating closeness of {0}d {1}".format(dimension, graphName))
        propertyMap = graph.new_vertex_property("double")
        closeness(graph, vprop=propertyMap)
        graph.vertex_properties[propertyName] = propertyMap
        graph.is_dirty = True
    return graph.vertex_properties[propertyName]



def saveGraph(graph, minDim, maxDim, outputDirectory):
    graphFileName = "graph_{0}-{1}d.xml.gz".format(minDim, maxDim)
    graph.save(outputDirectory + graphFileName)
    log.info("graph saved as " + outputDirectory + graphFileName)


def generateGraph(graphInputDirectory, graphFilenamePattern, positionFile, outputDirectory, graphNames, minDim, maxDim):
    log.info("loading positions")
    if positionFile.endswith("npz"):
        positions = np.load(positionFile)
    else:
        positions = np.genfromtxt(positionFile)

    graph = gen_graph_with_2dpositions(positions)

    log.info("adding proximity graphs for dimensions {0}-{1}".format(minDim, maxDim))
    # for dimensions a -> z
    for dim in range(minDim, maxDim + 1):
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
        for it in range(skip_header + 1):
            print("skipping header row {0}".format(it))
            f.readline()
        num_cols = len(f.readline().split(delimiter))
        print("num cols found: {0}\tdelimiter: {1}".format(num_cols, delimiter))
        # print("cols: {0}".format(num_cols))
    return np.genfromtxt(filename,
                         dtype=dtype, delimiter=delimiter, usecols=range(skip_column, num_cols),
                         skip_header=skip_header)
