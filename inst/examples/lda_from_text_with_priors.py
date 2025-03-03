import jpype, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    load_lda_dataset,
    new_simple_lda_config,
    print_top_words,
    sample_pclda,
)
# Define necessary functions like new_simple_lda_config, load_lda_dataset, etc.


java_home = '/usr/lib/jvm/java-11-openjdk'
classpath = 'inst/java/pcldar.jar'  # Optional: Set if you're using specific jars

# Start the JVM if not already running
if not jpype.isJVMStarted():
    jpype.startJVM(jpype.getDefaultJVMPath(), "-Djava.class.path=" + classpath)


# Configuration for LDA
nr_topics = 20
iterations = 1000
ds_fn = "na"  # This is a placeholder; you'll want to adjust this for the dataset path
cnf = new_simple_lda_config(ds_fn,
                             nr_topics=nr_topics, alpha=0.01,
                             beta=(nr_topics / 50), iterations=iterations,
                             rareword_threshold=10,
                             stoplist_fn="inst/examples/stoplist.txt",  # Path to stoplist.txt
                             topic_interval=10,
                             tmpdir="/tmp",
                             topic_priors = "inst/examples/ap_prior.txt" # path to seeds
                             )

# Load the dataset (replace with actual file path)
ds_fn = "inst/extdata/100ap.txt"  # Replace with actual file path
with open(ds_fn, 'r', encoding="latin1") as f:
    doclines = [_.lower() for _ in f.readlines()]

# Prepare the dataset
trtextdf = pd.DataFrame(doclines, columns=["line"])
trtextdf['line'] = trtextdf['line'].apply(lambda x: x.encode('ascii', errors='ignore').decode())
print(trtextdf)

# Create dataset for LDA
dss = create_lda_dataset(doclines, doclines, stoplist_fn="inst/examples/stoplist.txt")
print(dss)
lda = sample_pclda(cnf, dss['train'], iterations=iterations, testset=dss['test'])

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

print(ll, hll)
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
