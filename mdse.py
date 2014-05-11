import os

__author__ = 'nmew'

import sys

sys.path.insert(0, '/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages')

import csv
import logging
import numpy as np

from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

from scipy import stats

log = None


def main():
    parser = arguments()
    params = parser.parse_args(sys.argv)
    logging.basicConfig(stream=params.log)

    global readmeFilename
    readmeFilename = 'readme.txt'

    global log
    log = logging.getLogger('mdse')
    log.setLevel(logging.DEBUG)

    inputFile = params.dissimilarityInputFile           # input file containing dissimilarities
    outputDirectory = params.outputDirectory            # output directory to store embeddings and result summary
    embeddingComparator = params.embeddingComparator    # chisquared or stress to evaluate loss of embedding
    logFile = params.logFile                            # log file to write to
    minDimensions = params.minDimensions                # min dimensions to scale to
    maxDimensions = params.maxDimensions                # max dimensions to scale to
    skipHeader = params.skipHeader                      # skip this many rows in input file
    testSize = params.test                              # size of test dissimilarity matrix
    jobs = params.jobs                                   # number of processes to use in mds
    log.info("** Dissimilarity Graph Analysis **\n")

    # generate dissimilarity if test is set
    if testSize > 0:
        log.info("Generating dissimilarity of size {0}".format(testSize))
        dissimilarity = gen_fake_dissimilarities(testSize)
        log.info("Done Generating. Shape: {0}".format(dissimilarity.shape))
    # otherwise, read from input
    else:
        log.info("Reading dissimilarity from: {0}".format(inputFile))
        dissimilarity = genfromtxt(inputFile,
                                   skip_header=skipHeader, skip_column=skipHeader, dtype=np.float64, delimiter=params.delimiter.decode('string-escape'))
        log.info("Done Reading. Shape: {0}".format(dissimilarity.shape))

    bestStress, bestDim, bestSimilarity = find_best_mds(dissimilarity, maxDimensions, minDimensions, embeddingComparator,
                                                        logFile, outputDirectory, jobs)
    # output to readmed
    with open(outputDirectory + readmeFilename, 'a+') as f:
        f.write("\n Dimension with least loss: {0}\n({1})".format(bestDim, bestSimilarity))


def find_best_mds(dissimilarity, highest_dim, lowest_dim, embeddingComparator, logFile, outputDirectory, jobs):
    moleculeCount = molecule_count(dissimilarity)
    bestStress = None
    bestSimilarity = None
    bestDim = None
    for dim in range(lowest_dim, highest_dim + 1):
        isBest = False
        if embeddingComparator == 'chisquared':
            similarity, embedding, stress = compare_mds_per_mol_distance(dissimilarity, moleculeCount, dim, jobs, log)
            isBest = bestSimilarity is None or similarity > bestSimilarity

        # elif embeddingComparator == 'stress':
        else:
            # run mds on input file
            embedding, stress = mds(dissimilarity, dim, jobs, log)
            similarity = stress
            isBest = bestSimilarity is None or similarity < bestSimilarity

        logText = "{1}: similarity: {0}\tstress:{2} {3}" \
            .format(similarity, dim, stress, '(BEST)' if isBest else '')
        log.info(logText)
        log_to_file(logFile, logText)
        if isBest:
            bestSimilarity = similarity
            bestStress = stress
            bestDim = dim
        output_mds(embedding, similarity, dim, embeddingComparator, outputDirectory)

    logMessage = "#######################\nbestDim: {0}\t\tbestSimilarity: {1}\t\tbestStress: {2}\t\trangeTested: {3}\n" \
        .format(bestDim, bestSimilarity, bestStress, (highest_dim, lowest_dim))
    log.info(logMessage)
    log_to_file(logFile, logMessage)
    return bestStress, bestDim, bestSimilarity


def output_mds(embedding, similarity, dim, embeddingComparator, outputDirectory):
    readme = outputDirectory + readmeFilename
    # create dir if not already there
    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)
        with open(readme, 'a+') as f:
            f.write("Directory contains the coordinates from mds. Filename, Dimensions, Similarity ({0}):\n".format(
                embeddingComparator))
            # filename is [dim]_coordinates
    filename = '{0}_coordinates'.format(dim)
    # append filename, dimension and similarity to readme
    with open(readme, 'a') as f:
        f.write("{0}\t{1}\t{2}\n".format(filename, dim, similarity))

    log.info("writing file dim: {0}".format(dim))
    # write coordinates to file
    np.save(outputDirectory + filename, embedding)
    # np.savetxt(outputDirectory + filename + '.txt', embedding)


def compare_mds_all_distances(dissimilarity, moleculeCount, dimensions, log):
    # run mds on input file
    embedding, stress = mds(dissimilarity, dimensions, log)
    # calculate euclidean distances of exported coordinates
    distances = euclidean_distances(embedding)
    # histogram distances
    distHist, distHistEdges = np.histogram(distances, moleculeCount)
    # histogram dissimilarities
    dismHist, dismHistEdges = np.histogram(dissimilarity, moleculeCount)
    chisquare = stats.chisquare(distHist, dismHist)
    dismHist_distHist = compare_histograms(dismHist, distHist)
    return dismHist_distHist, chisquare


def compare_mds_per_mol_distance(dissimilarity, moleculeCount, dimensions, jobs, log):
    # run mds on input file
    embedding, stress = mds(dissimilarity, dimensions, jobs, log)
    # calculate euclidean distances of exported coordinates
    distances = euclidean_distances(embedding)
    # normalize them from 0 to 1
    normalizedDistances = normalize(distances)
    distHist = np.empty((moleculeCount, moleculeCount))
    normDistHist = np.empty((moleculeCount, moleculeCount))
    dissimilarityHist = np.empty((moleculeCount, moleculeCount))

    sum_dismHist_distHist = 0
    # sum_dismHist_normDistHist = 0
    # sum_distHist_normDistHist = 0
    # sum_distHist_distHist = 0

    for i in range(0, moleculeCount):
        # log.info("i:{0}".format(i))
        # histogram distances
        distHist[i] = np.histogram(distances[i], moleculeCount)[0]
        # histogram normalized distances
        normDistHist[i] = np.histogram(normalizedDistances[i], moleculeCount)[0]
        # histogram dissimilarities
        dissimilarityHist[i] = np.histogram(dissimilarity[i], moleculeCount)[0]

        # compare histograms and add to sum
        # sum_dismHist_distHist += math.log10(max(sys.float_info.min, compare_histograms(dissimilarityHist[i], distHist[i])[1]))
        sum_dismHist_distHist += stats.chisquare(dissimilarityHist[i], distHist[i])

    dismHist_distHist = sum_dismHist_distHist / moleculeCount
    return dismHist_distHist, embedding, stress


def mds(dissimilarity, dimensions, jobs, log):
    log.info("Creating MDS")
    # metric mds dimensions=3, SMACOF threshold of 100 iterations or 0.001 relative change (there is no time-limit)
    # mds = manifold.MDS(metric=True, verbose=2, dissimilarity="precomputed", n_components=dimensions, n_jobs=-2, max_iter=100, eps=0.001)

    # metric mds dimensions=3, defaults:
    mmds = manifold.MDS(metric=False, verbose=1, dissimilarity="precomputed", n_components=dimensions, max_iter=500,
                       eps=0, n_jobs=jobs)
    mmds.fit(dissimilarity)
    embedding = np.copy(mmds.embedding_)
    stress = mmds.stress_
    # del mmds
    # log.info("Fitting mds to dissimilarity")
    return embedding, stress


def genfromtxt(filename, skip_header=0, skip_column=0, dtype=np.float32, delimiter=','):
    with open(filename) as f:
        num_cols = len(f.readline().split(delimiter))
        # print("cols: {0}".format(num_cols))
    return np.genfromtxt(filename,
                         dtype=dtype, delimiter=delimiter, usecols=range(skip_column, num_cols),
                         skip_header=skip_header)


def gen_fake_dissimilarities(n_samples=50):
    seed = np.random.RandomState(seed=3)
    X_true = seed.randint(0, 20, 2 * n_samples).astype(np.float)
    X_true = X_true.reshape((n_samples, 2))
    # Center the data
    X_true -= X_true.mean()
    return euclidean_distances(X_true)


def output_coordinates(filename, data, stress=None):
    with open(filename, 'w+b') as f:
        writer = csv.writer(f)
        if stress is not None:
            writer.writerow(["stress", stress])
        for xyz in data:
            writer.writerow(xyz)


def log_to_file(filename, log):
    with open(filename, 'a+') as f:
        f.write(log + "\n")


def molecule_count(dissimilarity):
    rows, cols = dissimilarity.shape
    return cols


def compare_histograms(model, test):
    m, t = model, test
    minHist = np.minimum(m, t)
    # print(minHist)
    intersection = np.sum(minHist)
    normIntersection = float(intersection) / np.sum(model)
    # log.info("intersection {0}".format((intersection, normIntersection)))
    return intersection, normIntersection


def arguments():
    # argparse requires python 2.7
    from argparse import ArgumentParser
    from argparse import FileType
    #  "Run MDS on input distance file and save to output file"
    parser = ArgumentParser(prog="Dissimilarity Graph Analysis")
    parser.add_argument('--version', action='version', version='%(prog)s 0.1')
    parser.add_argument('pythonScript', metavar='pythonScript')
    parser.add_argument('dissimilarityInputFile',
                        type=str,
                        help='Path to dissimilarity file')
    parser.add_argument('outputDirectory',
                        type=str,
                        default='output/defaultDir/',
                        help='Path to output directory.')
    parser.add_argument('-minDim', '--minDimensions',
                        type=int,
                        default=2,
                        nargs='?',
                        help='Smallest dimension to reduce to.')
    parser.add_argument('-maxDim', '--maxDimensions',
                        type=int,
                        default=95,
                        nargs='?',
                        help='Smallest dimension to reduce to. If not set, calculated from input size.')
    parser.add_argument('-skipHeader', '--skipHeader',
                        type=int,
                        default=0,
                        nargs='?',
                        help='Leading columns and rows in the input file will be skipped')
    parser.add_argument('-c', '--embeddingComparator',
                        type=str,
                        default='stress',
                        nargs='?',
                        help='Measure to use to compare MDS embedding and determine dimension with least loss')
    parser.add_argument('-del', '--delimiter',
                        type=str,
                        default=',',
                        nargs='?',
                        help='Delimiter for input file')
    parser.add_argument('-j', '--jobs',
                        type=int,
                        default=1,
                        nargs='?',
                        help='The number of jobs to use for the computation. This works by breaking down the pairwise '
                             'matrix into n_jobs even slices and computing them in parallel.\n If -1 all CPUs are used.'
                             ' If 1 is given, no parallel computing code is used at all, which is useful for debugging'
                             ' For n_jobs below -1, (n_cpus + 1 + n_jobs) are used.'
                             ' Thus for n_jobs = -2, all CPUs but one are used.')
    parser.add_argument('-t', '--test',
                        type=int,
                        const=1000,
                        default=0,
                        nargs='?',
                        help='Uses a randomly generated dissimilarity matrix for input. Size of matrix is input^2.')
    parser.add_argument('-log', '--log',
                        type=FileType,
                        default=sys.stderr,
                        nargs='?',
                        help='Path to log errors [default: %(default)s]')
    parser.add_argument('-logFile', '--logFile',
                        type=str,
                        default='output/log.txt',
                        nargs='?',
                        help='Path to log file')
    # todo: connect log-level
    parser.add_argument('-log-level', '--log-level',
                        default='debug',
                        choices=['debug', 'info', 'warn'],
                        nargs='?',
                        help="Set log level (%(choices)s) [default: %(default)s]")
    return parser


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error: {0}".format(e))
        raise
        # return -1