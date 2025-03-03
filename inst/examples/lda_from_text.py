"""
Run a basic PCLDA on a test dataset
"""
import jpype, os
import pandas as pd
import numpy as np
from jpype import JClass
from pcldapy.pclda import (
    calculate_ttm_density,
    create_lda_dataset,
    get_held_out_log_likelihood,
    get_log_likelihood,
    get_phi,
    get_theta_estimate,
    get_top_relevance_words,
    get_topwords,
    get_type_topics,
    get_z_means,
    load_lda_config,
    load_lda_dataset,
    new_simple_lda_config,
    print_top_words,
    sample_pclda,
)
import argparse
import matplotlib.pyplot as plt
import os



def ensure_jvm(jar):
    # Start the JVM if not already running
    if not jpype.isJVMStarted():
        #jpype.startJVM(jpype.getDefaultJVMPath(), "-Djava.class.path=" + classpath)
        jpype.startJVM(classpath=[jar])

def main(args):

    if args.new_config:
        # The first time, you'll want to set up a config file, with essential variables

        if args.jar_path is None:
            args.jar_path = "/home/bob/Installers/PartiallyCollapsedLDA/target/PCPLDA-9.2.2-jar-with-dependencies.jar"

        ensure_jvm(args.jar_path)

        jars = {args.jar_name: args.jar_path}
        cnf = new_simple_lda_config(
                    dataset = args.dataset,
                    nr_topics = args.n_topics,
                    alpha = args.alpha,
                    beta = (args.n_topics / 50),
                    iterations = args.iterations,
                    rareword_threshold = args.rareword_threshold,
                    stoplist_fn = args.stoplist,
                    topic_interval = args.topic_interval,
                    tmpdir = args.tmpdir,
                    jar_dict = jars,
                    cfg_fn = args.config)
    else:
        #otherwise, you can load an existing config file
        cnf, jars = load_lda_config(args.config)
        ensure_jvm(jars[args.jar_name])


    # Load the dataset (replace with actual file path)
    with open(str(cnf.getDatasetFilename()), 'r', encoding="latin1") as f:
        doclines = f.readlines()

    # Prepare the dataset
    trtextdf = pd.DataFrame(doclines, columns=["line"])
    trtextdf['line'] = trtextdf['line'].apply(lambda x: x.encode('ascii', errors='ignore').decode())
    print(trtextdf)

    # Create dataset for LDA
    dss = create_lda_dataset(doclines, doclines, stoplist_fn=str(cnf.getStoplistFilename()))#"inst/examples/stoplist.txt")
    print(dss)
    lda = sample_pclda(cnf, dss['train'], iterations=args.iterations, testset=dss['test'])

    # Extract results
    phi = get_phi(lda)
    ttm = get_type_topics(lda)
    dens = calculate_ttm_density(ttm)
    zBar = get_z_means(lda)
    theta = get_theta_estimate(lda)

    # Get top words and top relevance words
    tw = get_topwords(lda)
    trw = get_top_relevance_words(lda, cnf)

    # Log-likelihood and held-out log-likelihood
    ll = get_log_likelihood(lda)
    hll = get_held_out_log_likelihood(lda)

    # Prepare the statistics for plotting
    stats = pd.DataFrame({
        'iter': np.arange(1, len(ll) + 1) * 10,
        'loglikelihood': ll,
        'heldout_likelihood': hll
    })

    # Reshape the stats for plotting
    stats_melted = pd.melt(stats, id_vars=["iter"], value_vars=["loglikelihood", "heldout_likelihood"])

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(stats_melted['iter'], stats_melted['value'], label="Log-Likelihood & Held-Out Log-Likelihood", color='blue')
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.title('Log-Likelihood and Held-Out Log-Likelihood over Iterations')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print the top words
    print("Top Words:")
    print_top_words(get_topwords(lda))

    # Print the top relevance words
    print("Top Relevance Words:")
    print_top_words(get_top_relevance_words(lda, cnf))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=0.01, help="alpha")
    parser.add_argument("--config",
                        type=str,
                        default=None,
                        help="path to config file. When creating a new config, it will be written at this path if provided. Otherwise, this is good for reading an existing config file.")
    parser.add_argument("--dataset", type=str, help="path to dataset")
    parser.add_argument("-i", "--iterations", type=int, default=1000, help="Number of iterations")
    parser.add_argument("--jar-name", default="default", help="Shorthand name of a PCLDA jar file.")
    parser.add_argument("--jar-path", default=None, help="path to PCLDA jar file")
    parser.add_argument("-n","--n-topics", type=int, default=20, help="Number of topics")
    parser.add_argument("--new-config", action='store_true', help="set this flag to create a new config file")
    parser.add_argument("-r", "--rareword_threshold", type=int, default=10, help="rare word threshold")
    parser.add_argument("-s", "--stoplist", default=None, type=str, help="path to stoplist file")
    parser.add_argument("-t", "--topic-interval", type=int, default=10, help="topic interval")
    parser.add_argument("--tmpdir", default="/tmp", help="temporary directory")
    args = parser.parse_args()
    main(args)
