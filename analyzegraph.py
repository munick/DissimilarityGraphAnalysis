__author__ = 'nmew'
__version__ = 0.1
import sys
sys.path.insert(0,'/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages')
import logging
import numpy as np
from graph_tool.all import *

def main():
    parser = arguments()
    # single dimension
    # params = parser.parse_args(['', 'output/test/{0}_graphs/', 'output/test/', '-pos', 'output/test/2_coordinates.npy'])
    # range of dimensions
    # params = parser.parse_args(['', 'output/test/{0}_graphs/', 'output/test/', '-pos', 'output/test/2_coordinates.npy', '-d', '2', '9'])

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
    # for dim in range(minDim, maxDim+1):
    #     outputProxiGraphOverlaps(graph, ['gg', '2nn'], dim, outputDirectory + "rng_gg_{0}d_overlap.png".format(minDim))


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