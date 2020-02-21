import numpy as np
import ndjson

# directory where the raw data is stored
data_dir = "data"

# file extension for data files
file_ext = ".ndjson"

# object classes we are interested in, these should match the file names
classes = ["circle", "square", "hexagon"]

# ratio for how training / test datasets
trainingSetRatio = 0.2

# holder for full dataset
data = {}

# index values of training and test data elements
trainingIndices = {}
testIndicies = {}

# load data
for i, c in enumerate(classes):
    with open(data_dir + "/" + classes[i] + file_ext) as f:

        # load only data where we know it was recognized in the dataset
        d = []
        for line in f:
            content = ndjson.loads(line)
            if (content[0]['recognized'] is True):
                d.append(content[0])
        data[c] = d

        l = len(d)

        # create indicies for training and test data sets
        trainingIndices[c] = np.random.randint(0, l, int(trainingSetRatio * l))
        testIndicies[c] = np.arange(0,l)
        testIndicies[c] = np.delete(testIndicies[c], trainingIndices[c])
