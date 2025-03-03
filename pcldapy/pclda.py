import gc
import jpype
import json
import os
import sys
import warnings



def write_config(slc, cfg_fn, jar_dict=None):
    """
    Take a config object and a file name, write the config to a json file

    Args

        slc: config onbject
        cfg_fn
    """
    def _custom_serializer(obj):
        """
        This takes an object and serializes it into a string if it can't be done automatically.
        """
        try:
            return json.JSONEncoder().default(obj)
        except TypeError:
            try:
                return vars(obj)['_jstr']
            except Exception as e:
            #    print(f"Non-serializable: {obj}, {vars(obj)} {vars(obj.__class__)} :: {e}\n\n")
                return f"Non-serializable: {type(obj).__name__}"

    defaults = {
        "VariableSelectionPrior": 0.5,
        "TopicInterval": 10,
        "TfIdfVocabSize": -1,
        "StoplistFilename": "stoplist.txt",
        "StartDiagnostic": 500,
        "Seed": -1,
        "SavedSamplerDirectory": "stored_samplers",
        "ResultSize": 1,
        "PhiMeanThin": 1,
        "PhiBurnInPercent": 0,
        "NrTopWords": 20,
        "NoTopicBatches": 2,
        "NoIterations": 1500,
        "NoBatches": 4,
        "MaxDocumentBufferSize": 10000,
        "Lambda": 0.6,
        "KeepConnectingPunctuation": False,
        "IterationCallbackClass": None,
        "HyperparamOptimInterval": -1,
        "HDPNrStartTopics": 1,
        "HDPKPercentile": 0.8,
        "HDPGamma": 1,
        "FileRegex": ".*\\.txt$",
        "DocumentSamplerSplitLimit": 100,
        "Beta": 0.01,
        "Alpha": 50 / slc.getNoTopics()
    }

    skip_keys = [
    #    "DatasetFilename",
        "DocumentBatchBuildingScheme",
        "ExperimentOutputDirectory",
        "FullPhiPeriod",
        "InstabilityPeriod",
        "IntArrayProperty",
        "LoggingUtil",
        "PrintNDocsInterval",
        "PrintNTopWordsInterval",
        "SubConfigs",
        "TopicBatchBuildingScheme",
        "TopicIndexBuildingScheme",
        "SubTopicIndexBuilders",
        "LogUtil",
    ]

    slc_out = {}
    for k, v in slc.__class__.__dict__.items():
        #if "lpha" in k:
        #    print("!!!!!~~~~~", k)
        if k.startswith("get"):
            E = []
        #    print(k)
            if k[3:] not in skip_keys:
                try:
                    slc_out[k[3:]] = getattr(slc,k,None)()
                except Exception as e:
                    E.append(f"  calling {k} didnt work: {e}")
                    try:
                        E.append(f"  -> trying w/ default values")
                        slc_out[k[3:]] = getattr(slc,k,None)(defaults[k[3:]])

                    except Exception as e:
                        pass
        #                E.append(f" !! oh no: {e}")
        #        try:
        #            print("  --", slc_out[k[3:]])
        #            print("  ---- OK")
        #        except:
        #            [print(e) for e in E]

    jars_out = {}
    with open(cfg_fn, 'r') as oldconfig:
        oslc = json.load(oldconfig)
        if "jars" in oslc:
            jars_out = oslc["jars"]

    for k, v in jar_dict.items():
        if k is not None:
            jars_out[k] = v

    cfg_out = {"pclda_config": slc_out, "jars": jars_out}
    with open(cfg_fn, "w+") as outf:
        json.dump(cfg_out, outf, default=_custom_serializer, ensure_ascii=False, indent=4)


def new_simple_lda_config(
        dataset="dataset.txt",
        nr_topics=20,
        alpha=None,
        beta=None,
        iterations=2000,
        rareword_threshold=10,
        optim_interval=-1,
        stoplist_fn="stoplist.txt",
        topic_interval=10,
        tmpdir="/tmp",
        topic_priors="priors.txt",
        jar_dict = {},
        cfg_fn = None
    ):
    """
    Create a new LDA config file with default values, unless otherwise specified.

    Args

        dataset_fn: filename of dataset (in LDA format)
        nr_topics: number of topics to use
        alpha: symmetric alpha prior
        beta: symmetric beta prior
        iterations: number of iterations to sample
        rareword_threshold: min. number of occurences of a word to be kept
        optim_interval: how often to do hyperparameter optimization (default is off = -1)
        stoplist_fn: filenname of stoplist file (one word per line) (default "stoplist.txt")
        topic_interval: how often to print topic info during sampling
        tmpdir: temporary directory for intermediate storage of logging data (default "tmp")
        topic_priors: text file with 'prior spec' with one topic per line with format: <topic nr(zero idxed)>, <word1>, <word2>, etc
        jar_dict: named jar file dict. If you only work with on jar file this should be `{'default': 'path/to/jarfile'}`
        cfg_fn: path to config file. if the file exists, it will be updated with provided values, if not the new config will be written to a json file.

    Returns

        config object
    """

    if alpha is None:
        alpha = 50 / nr_topics

    if beta is None:
        beta = nr_topics / 5

    # Initialize LoggingUtils
    lu = jpype.JClass("cc.mallet.util.LoggingUtils")()
    lu.checkAndCreateCurrentLogDir(tmpdir)

    # Initialize SimpleLDAConfiguration
    slc = jpype.JClass("cc.mallet.configuration.SimpleLDAConfiguration")()

    # Set the LoggingUtil
    slc.setLoggingUtil(lu)

    # Set other parameters
    slc.setNoTopics(nr_topics)
    slc.setAlpha(jpype.JDouble(alpha))
    slc.setBeta(jpype.JDouble(beta))
    slc.setNoIters(jpype.JInt(iterations))
    slc.setRareThreshold(jpype.JInt(rareword_threshold))
    slc.setTopicInterval(jpype.JInt(topic_interval))
    slc.setStartDiagnostic(jpype.JInt(90))
    slc.setNoBatches(jpype.JInt(5))
    slc.setStoplistFilename(stoplist_fn)
    slc.setTopicPriorFilename(topic_priors)
    slc.setDatasetFilename(dataset)
    slc.setHyperparamOptimInterval(jpype.JInt(topic_interval))
    slc.setNoPreprocess(True)

    print(len(jar_dict))
    if len(jar_dict) == 0:
        warnings.warn("You need at least one PCLDA jarfile to work with this library, but you haven't provided one.")
        inp = input("Do you want to provide a default PCLDA jar file now? Enter the path from the cwd (or q to exit): ")
        if inp == 'q':
            print("Ok, exiting")
            sys.exit()
        else:
            jar_dict = {"default": os.path.abspath(inp)}

    if cfg_fn is not None:
        write_config(slc, cfg_fn, jar_dict=jar_dict)

    return slc


def load_lda_config(cfg_fn):
    """
    Load the lda config file from a json file.

    Args

        cfg_fn: path to json config file

    Returns

        config object
    """
    java_types = {
            "Alpha": jpype.JDouble,
            "Beta": jpype.JDouble,
            "HyperparamOptimInterval": jpype.JInt,
            "NoBatches": jpype.JInt,
            "NoIters": jpype.JInt,
            "NoTopicBatches": jpype.JInt,
            "RareThreshold": jpype.JInt,
            "StartDiagnostic": jpype.JInt,
            "TfIdfThreshold": jpype.JInt,
            "TopicInterval": jpype.JInt,
        }
    # load jaon cfg declaration  as a dict
    with open(cfg_fn, 'r') as inf:
        j = json.load(inf)

    if "Beta" not in j or j["Beta"] is None:
        j["Beta"] = j["NoTopics"] / 50

    # Initialize SimpleLDAConfiguration
    slc = new_simple_lda_config(jar_dict=j["jars"])

    # replace slc init values with dict
    for k, v in j["pclda_config"].items():
        method_name = f"set{k}"
        method = getattr(slc, method_name, None)
        if method:
            if k in java_types:
                method(java_types[k](v))
            else:
                method(v)
        else:
            warnings.warn(f"Unrecognized key in provided config file :: {k} = {v}")
    return slc, j["jars"]


def load_lda_dataset(fn, ldaconfig):
    # Initialize LDAUtils
    util = jpype.JClass("cc.mallet.util.LDAUtils")()

    # Call the loadDataset method
    ds = util.loadDataset(jpype.JObject(ldaconfig, "cc.mallet.configuration.LDAConfiguration"), fn)

    return ds


def load_lda_sampler(ldaconfig, ds, store_dir="stored_samplers"):
    # Initialize LDAUtils
    util = jpype.JClass("cc.mallet.util.LDAUtils")()

    # Load the stored sampler
    ss = util.loadStoredSampler(ds, jpype.JObject(ldaconfig, "cc.mallet.configuration.LDAConfiguration"), store_dir)

    # Get the sampler type
    sampler_type = ss.getClass().getName()

    # Create the new sampler
    lcfg = jpype.JObject(ldaconfig, "cc.mallet.configuration.LDAConfiguration")

    try:
        lda = jpype.JClass(sampler_type)(lcfg)
    except jpype.JException as ex:
        lda = ex  # Capture the exception if creation fails

    # Init the new sampler from the stored sampler
    lda_instance = jpype.JObject(lda, "cc.mallet.topics.LDASamplerInitiable")
    lda_instance.initFrom(jpype.JObject(ss, "cc.mallet.topics.LDAGibbsSampler"))

    # Return the new initialized sampler
    return lda


def create_lda_dataset(train, test=None, stoplist_fn="stoplist.txt"):
    # Initialize StringClassArrayIterator for the training data
    string_iterator = jpype.JClass("cc.mallet.util.StringClassArrayIterator")(train)

    # Initialize LDAUtils and create a Pipe
    util = jpype.JClass("cc.mallet.util.LDADatasetStringLoadingUtils")()
    #pipe = util.buildSerialPipe(stoplist_fn, jpype.JNull("cc.mallet.types.Alphabet"), True)
    pipe = util.buildSerialPipe(stoplist_fn, None, True)
    print(pipe)
    # Create InstanceList for the training data
    il = jpype.JClass("cc.mallet.types.InstanceList")(pipe)
    il.addThruPipe(jpype.JObject(string_iterator, "java.util.Iterator"))

    # If test data is provided, process it similarly
    if test is not None:
        string_iterator_test = jpype.JClass("cc.mallet.util.StringClassArrayIterator")(test)

        # Get the alphabet from the training data
        train_alphabet = il.getAlphabet()

        # Create a Pipe for the test data
        test_pipe = util.buildSerialPipe(stoplist_fn, train_alphabet, True)

        # Create InstanceList for the test data
        iltest = jpype.JClass("cc.mallet.types.InstanceList")(test_pipe)
        iltest.addThruPipe(jpype.JObject(string_iterator_test, "java.util.Iterator"))

        return {'train': il, 'test': iltest}

    return il


def sample_pclda(ldaconfig, ds, iterations=2000, sampler_type="cc.mallet.topics.PolyaUrnSpaliasLDA",
                 testset=None, save_sampler=True):
    # Cast the LDA configuration
    lcfg = jpype.JObject(ldaconfig, "cc.mallet.configuration.LDAConfiguration")

    # Try to create the LDA sampler using the specified type
    try:
        lda = jpype.JClass(sampler_type)(lcfg)
    except jpype.JException as ex:
        lda = ex  # Handle the exception if the sampler creation fails

    # Add instances to the sampler
    lda.addInstances(ds)

    # If test data is provided, add it to the sampler
    if testset is not None:
        lda.addTestInstances(testset)

    # Perform sampling
    lda.sample(iterations)

    # If we need to save the sampler, perform the saving procedure
    if save_sampler:
        sampler_dir = jpype.JClass("cc.mallet.configuration.LDAConfiguration").STORED_SAMPLER_DIR_DEFAULT
        sampler_folder = ldaconfig.getSavedSamplerDirectory(sampler_dir)

        util = jpype.JClass("cc.mallet.util.LDAUtils")()
        util.saveSampler(jpype.JObject(lda, "cc.mallet.topics.LDAGibbsSampler"),
                         jpype.JObject(ldaconfig, "cc.mallet.configuration.LDAConfiguration"),
                         sampler_folder)

    return lda


def sample_pclda_continue(lda, iterations=2000):
    # Perform additional sampling
    lda.sample(iterations)

    return lda


def print_top_words(word_matrix):
    """
    # Initialize LDAUtils
    util = jpype.JClass("cc.mallet.util.LDAUtils")()

    # Call the formatTopWords method and return the result
    return util.formatTopWords(jpype.JArray(jpype.JDouble)(word_matrix))
    """
    # Initialize LDAUtils
    util = jpype.JClass("cc.mallet.util.LDAUtils")()

    # Create a 2D Java array (an array of arrays of JString)
    word_matrix_java = jpype.JArray(jpype.JArray(jpype.JString))(len(word_matrix))

    for i, row in enumerate(word_matrix):
        # Convert each row to a JArray of JString, passing the whole row as a single argument
        word_matrix_java[i] = jpype.JArray(jpype.JString)(row)

    result = util.formatTopWords(word_matrix_java)

    # Print or return the result to check what it's giving us
    print("Formatted top words result:", result)
    return result


def extract_vocabulary(alphabet):
    # Initialize LDAUtils
    util = jpype.JClass("cc.mallet.util.LDAUtils")()

    # Call the extractVocabulary method and return the result
    return util.extractVocabulary(alphabet)


def extract_term_counts(instances):
    # Initialize LDAUtils
    util = jpype.JClass("cc.mallet.util.LDAUtils")()

    # Call the extractTermCounts method and return the result
    return util.extractTermCounts(instances)


def extract_doc_lengths(instances):
    # Initialize LDAUtils
    util = jpype.JClass("cc.mallet.util.LDAUtils")()

    # Call the extractDocLength method and return the result
    return util.extractDocLength(instances)


def get_alphabet(lda):
    # Call the getAlphabet method on the lda object and return the result
    return lda.getAlphabet()


def get_theta_estimate(lda):
    # Call the getThetaEstimate method on the lda object and return the result
    theta = lda.getThetaEstimate()
    return theta

def get_z_means(lda):
    # Call the getZbar method on the lda object and return the result
    zb = lda.getZbar()
    return zb


def get_type_topics(lda):
    # Call the getTypeTopicMatrix method on the lda object and return the result
    ttm = lda.getTypeTopicMatrix()
    return ttm


def get_phi(lda):
    # Call the getPhi method on the lda object and return the result
    phi = lda.getPhi()
    return phi


def get_topwords(lda, nr_words=20):
    # Initialize LDAUtils
    util = jpype.JClass("cc.mallet.util.LDAUtils")()

    # Get the Alphabet from the LDA model
    alph = lda.getAlphabet()

    # Get the Type-Topic Matrix
    type_topic_matrix = lda.getTypeTopicMatrix()

    # Get the size of the alphabet and number of topics
    alph_size = alph.size()
    nr_topics = lda.getNoTopics()

    # Call the getTopWords method from LDAUtils
    tw = util.getTopWords(nr_words, alph_size, nr_topics, type_topic_matrix, alph)

    return tw


def get_top_relevance_words(lda, config, nr_words=20, lambda_value=0.6):
    # Initialize LDAUtils
    util = jpype.JClass("cc.mallet.util.LDAUtils")()

    # Get the Alphabet from the LDA model
    alph = lda.getAlphabet()

    # Get the Type-Topic Matrix
    type_topic_matrix = lda.getTypeTopicMatrix()

    # Get the size of the alphabet and number of topics
    alph_size = alph.size()
    nr_topics = lda.getNoTopics()

    # Get the beta value from the configuration
    beta_d = config.getBeta(0.01)
    beta = beta_d.doubleValue()

    # Call the getTopRelevanceWords method from LDAUtils
    rw = util.getTopRelevanceWords(nr_words, alph_size, nr_topics, type_topic_matrix, beta, lambda_value, alph)

    return rw


def calculate_ttm_density(type_topic_matrix):
    # Initialize LDAUtils
    util = jpype.JClass("cc.mallet.util.LDAUtils")()

    # Call the calculateMatrixDensity method and return the result
    return util.calculateMatrixDensity(type_topic_matrix)


def get_log_likelihood(lda):
    # Call the getLogLikelihood method on the lda object and return the result
    ll = lda.getLogLikelihood()
    return ll


def get_held_out_log_likelihood(lda):
    # Call the getHeldOutLogLikelihood method on the lda object and return the result
    ll = lda.getHeldOutLogLikelihood()
    return ll


def run_gc(*args):
    # Run Python's garbage collection
    gc.collect()

    # Call Java's garbage collection through Runtime.getRuntime().gc()
    jpype.JClass("java.lang.Runtime").getRuntime().gc()

    # No return value (invisible in R, so we return None in Python)
    return None

