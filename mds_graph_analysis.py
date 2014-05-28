import sys
import os
import subprocess

__author__ = 'nmew'
__version__ = 0.1


def main():
    parser = arguments()
    testArgs = "asdf -dissimilarityFile data/Pspace/dali.dist -dissimilarityMeasure dali -outputDirectory output/test_bundled/ --dimensions 2 4 --jobs -2".split()
    params = parser.parse_args(sys.argv)

    dissimilarityFilePath = os.path.abspath(params.dissimilarityFile)
    outputDirectory = os.path.abspath(params.outputDirectory)
    dissimilarityMeasure = params.dissimilarityMeasure
    jobs = params.jobs

    # set min and max dimensions
    minDim = params.dimensions[0]
    if len(params.dimensions) == 2:
        maxDim = params.dimensions[1]
    else:
        maxDim = minDim

    # create output dir if necessary
    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    embeddingFilePath = outputDirectory + "/{0}d_" + dissimilarityMeasure + "_embedding.txt"
    mdsJarPath = os.path.abspath("mdsjava.jar")

    # 1) run mds
    print("running mds")
    java_mds_command = "java -cp " + mdsJarPath + " edu.sfsu.mdsjava.MdsJava -i " + dissimilarityFilePath
    java_mds_command += " -o " + embeddingFilePath
    java_mds_command += " -d {0} {1}".format(minDim, maxDim)
    print("\tcalling " + java_mds_command)
    # run command and wait for completion
    subprocess.call(java_mds_command, shell=True)

    graphDirectory = outputDirectory + "/{0}d_graphs/"

    # 2) run graph creation
    print("creating proximity graphs")
    # create proximity graphs (gabriel, relative neighbor, nn1, nn2, nn3, nn4) for coordinates
    python_proximity_graph_command = "python graphmds.py {0} {1}".format(embeddingFilePath, graphDirectory)
    python_proximity_graph_command += " -b {0} {1} -j {2}".format(minDim, maxDim, jobs)
    print("\tcalling " + python_proximity_graph_command)
    subprocess.call(python_proximity_graph_command, shell=True)

    analysisDirectory = outputDirectory + "/analysis/"
    coordinatesFilePath = embeddingFilePath.format(2)
    # create output dir if necessary
    if not os.path.exists(analysisDirectory):
        os.mkdir(analysisDirectory)

    # 3) create graph analysis file
    print("creating analysis file")
    # create proximity graphs (gabriel, relative neighbor, nn1, nn2, nn3, nn4) for coordinates
    python_analysis_graph_command = "python create_analysis_graph.py " + graphDirectory + " " + analysisDirectory
    python_analysis_graph_command += " -pos " + coordinatesFilePath
    python_analysis_graph_command += " -d {0} {1}".format(minDim, maxDim)
    print("\tcalling " + python_analysis_graph_command)
    subprocess.call(python_analysis_graph_command, shell=True)

    # analysis graph dir
    analysisGraphFilePath = analysisDirectory + "graph_{0}-{1}d.xml.gz".format(minDim, maxDim)

    # 4) run graph analysis
    python_analysis_command = "python run_graph_analysis.py " + analysisGraphFilePath + " " + analysisDirectory
    python_analysis_command += " -d {0} {1} -nprops -intersect -btwn -deg -close".format(minDim, maxDim)
    print("\tcalling " + python_analysis_command)
    subprocess.call(python_analysis_command, shell=True)

    print("complete. Analysis files are located in: " + analysisDirectory)


def arguments():
    # argparse requires python 2.7
    from argparse import ArgumentParser
    from argparse import FileType
    #  "Run MDS on input distance file and save to output file"
    parser = ArgumentParser(prog="Dissimilarity Graph Analysis - Runs MDS -> Creates Graphs -> Analyzes Graphs",
                            version=__version__)
    parser.add_argument('pythonScript', metavar='pythonScript')
    parser.add_argument('-dissimilarityFile',
                        type=str,
                        help='Path to previously created and saved dissimilarity matrix.')
    parser.add_argument('-dissimilarityMeasure',
                        type=str,
                        help='Name of the dissimilarity measure used (eg. dali). This will be used for creating '
                             'directories so no spaces, slashes etc.')
    parser.add_argument('-outputDirectory',
                        type=str,
                        default='output/defaultDir/',
                        help='Path to output directory.')
    parser.add_argument('-d', '--dimensions',
                        type=int,
                        default=[2, 10],
                        nargs='*',
                        help='min and max dimensions for which to do embedding graph creation and analysis')
    parser.add_argument('-g', '--graphTypes',
                        type=str,
                        default=['gg', 'rng', '3nn', '2nn', '1nn'],
                        nargs='?',
                        help='List graph types to generate and analyze')
    parser.add_argument('-j', '--jobs',
                        type=int,
                        default=1,
                        nargs='?',
                        help='The number of jobs to use for the computation. This works by breaking down the pairwise '
                             'matrix into n_jobs even slices and computing them in parallel.\n If -1 all CPUs are used.'
                             ' If 1 is given, no parallel computing code is used at all, which is useful for debugging'
                             ' For n_jobs below -1, (n_cpus + 1 + n_jobs) are used.'
                             ' Thus for n_jobs = -2, all CPUs but one are used.')
    return parser


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error: {0}".format(e))
        raise
        # return -1