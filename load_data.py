# Utitlities to load quickdraw dataset

from os import path
import numpy as np
import cv2

# Load raw data and apply preprocessing
def load(classes, data_dir, file_ext=".npy", reload_data=False, samples=10, img_size=[448,448], num_cells=[7,7]):

    # web storage location for data files
    base_url = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'

    # data holders, x:input, y:output
    x = np.empty([0, img_size[0], img_size[1], 3])
    y = np.empty([0, num_cells[0], num_cells[1], (len(classes)+4+1)]) # 4: bounding box, 1: class probability
    labels = np.empty([0])

    # check if we need to perform full download
    if(not path.exists(data_dir) or reload_data):
        print('Loading full dataset')
    else:
        # load the data we need
        print('Loading data from folder: ' + data_dir)
        for idx, c in enumerate(classes):
            # download the data file if we don't already have it
            if(not path.exists(data_dir + "/" + c + file_ext)):
                download_data_file(c, base_url)

            # load raw data and grab only the number of samples we are to use
            data = np.load(data_dir + "/" + c + file_ext)
            data = data[0:samples, :, None, None]

            # process and prepare input data, add to data holder
            print('Processing input data for %s class...' % c)

            data_x = prepare_x(data, x.shape)
            x = np.concatenate((x, data_x), axis=0)

            # process and prepare response data, add to holder
            print('Processing response data for %s class...' % c)
            labels = np.full(x.shape[0], idx)
            l = np.append(y, labels)
            data_y = prepare_y(data_x, labels, classes, classes[idx], num_cells, y.shape)
            y = np.concatenate((y, data_y), axis=0)
        
        # Reshape the output to match loss function
        y = y.reshape(y.shape[0], y.shape[1]*y.shape[2]*y.shape[3])
        print('Final input data is %s' % str(x.shape))
        print('Final response data is %s' % str(y.shape))

    return (x,y)

def download_data_file(object_class, url):
    # we need to download this file
    print("Downloading %s data file" % object_class)
    cls_url = object_class.replace('_', '%20')
    urllib.request.urlretrieve(url + cls_url + file_ext, data_dir + '/' + object_class + file_ext)

    return None

# Resize and perform preprocesing on input data
def prepare_x(data, output_size):
    # Input: N x L
    # Output: N x w2 x h2 x 3
    # N: Number of samples
    # w, h: Initial image size
    # w2, h2: Output image size

    # make sure our input array is not empty
    assert (len(data) > 0), 'data is empty'

    # check to make sure we have the right size input
    assert (len(output_size) == 4), "Output size should have 4 dimensions: N (samples) x w (target image width) x h (target image height) x d (dimensions)"
    N1, L1, h1, d1  = data.shape
    N2, w2, h2, d2 = output_size
    s = int(np.sqrt(L1))

    # build output data array
    output = np.empty([0, w2, h2, d2])

    # resize and process all input for output
    for img in data:
        # enlarge images to expected input size
        img = img.reshape(s,s)
        img = cv2.resize(img, (w2, h2))

        # duplicate single channel to 3-dimensional
        img_out = np.empty([w2, h2, d2])
        for dim in range(d2):
            # normalize
            img = img.astype('float64')
            img /= 255.0
            img_out[:,:,dim] = img

        # add to output array
        output = np.concatenate((output, img_out.reshape(1, 448, 448, 3)), axis=0)

    return output

# resise and perform preprocessing on output responses
def prepare_y(data, labels, classes, this_class, num_cells, response_size):
    # Input: N x 1
    # Output: N x cw x ch  x (1 + number of classes + 4 (bounding box) + 1)
    # N: Number of samples
    # cw, ch: number of cells in width/height direction

    # make sure our intput array is not empty
    assert (len(data) > 0), "data is empty"

    # check to make sure we have the right size input
    assert (len(response_size) == 4), "Response size should have 4 dimensions: N (samples) x cw (horizontal cells) x ch (vertical cells) x v (output vector)"
    N, cw, ch, v = response_size

    # calculate output vector size per cell
    v = len(classes) + 1 + 4

    # create output response holder
    output = np.empty([0, num_cells[0] * num_cells[1], v])

    # generate response vectors for each image
    for idx, img in enumerate(data):
        img_output_vectors = np.empty([0, v])
        img_cells = split_image_into_cells(img, num_cells)

        # iterate over all cells and calculate the output vector
        for i in range(img_cells.shape[0]):

            # grab the cell
            cell = img_cells[i,:,:]
            
            # generate class specific probability
            class_values = np.zeros([len(classes)])
            class_idx = classes.index(this_class)
            class_values[class_idx] = 1

            # holder for bounding box values
            pc, bx, by, bw, bh = 0, 0, 0, 0, 0
            box_values = np.array([pc, bx, by, bw, bh])

            # update value for non-zero cells
            if not all(i == 0 for i in cell.flatten()):

                # assume class is present for non-zero values
                pc = 1
                
                # Grab all non-zero values' index 
                bx_index = np.where(cell!=0)[0]
                by_index = np.where(cell!=0)[1]

                # bw, bh: distence between start to last index
                bw = bx_index[len(bx_index) - 1] - bx_index[0]
                bh = by_index[len(by_index) - 1] - by_index[0]

                # bx, by: start index plus half width and height
                bx = bx_index[0] + bw // 2
                by = by_index[0] + bh // 2

            # update bounding box values
            box_values = [pc, bx, by, bw, bh]

            # concatenate bounding box and class values into one output vector
            output_vector = np.concatenate((box_values, class_values)).reshape(1,v)
            img_output_vectors = np.concatenate((img_output_vectors, output_vector), axis=0)

        # concatenate all output vectors for all cells in this image
        output = np.concatenate((output, img_output_vectors.reshape(1,num_cells[0] * num_cells[1],v)), axis=0)

    # reshape to output requirements
    output = output.reshape(len(data), num_cells[0], num_cells[1], v)

    return output

def split_image_into_cells(image, num_cells):
    # determine cell sizing
    ch = int(np.fix(image.shape[0]/num_cells[0]))
    cw = int(np.fix(image.shape[1]/num_cells[1]))

    # create cell holder
    cells = np.empty([0, cw, ch])

    # only interesteed pixel values, not colors
    if(image.ndim>2):
        image = image[:,:,0]

    # split up image matrix into cells
    for i in range(num_cells[0]):
        for j in range(num_cells[1]):
            # generate cell indicies
            h_idx = i*ch
            v_idx = j*cw

            cell = image[h_idx:h_idx+cw, v_idx:v_idx+ch]
            cells = np.concatenate((cells, cell.reshape(1,ch,cw)), axis=0)

    return cells

def divide_into_sets(input, response, ratio):
    # create indicies
    l = len(input)
    trainingIndices = np.random.randint(0, l, int(ratio * l))
    testIndicies = np.arange(0,l)
    testIndicies = np.delete(testIndicies, trainingIndices)

    # seperate data
    x_train = input[trainingIndices]
    y_train = response[trainingIndices]
    x_test = input[testIndicies]
    y_test = response[testIndicies]

    return x_train, y_train, x_test, y_test
