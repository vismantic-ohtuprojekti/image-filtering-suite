import numpy as np
import cv2
import math
from operator import itemgetter

from ..utils.statistic_common import linear_normalize
from ..utils.image_utils import read_color_image
from ..utils.utils import scaled_prediction

from ..utils.histogram_analyzation import normalize
from ..utils.histogram_analyzation import calculate_peak_value
from ..utils.histogram_analyzation import largest
from ..utils.histogram_analyzation import calculate_continuous_distribution
from ..utils.histogram_analyzation import calculate_derivates
from ..utils.histogram_analyzation import calc_standard_deviation
from ..utils.histogram_analyzation import remove_from_ends

from .. import get_data

from svm_filter import SVMFilter


def count_dispersion(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [180], [0,180])

    hist = remove_from_ends(hist)

    normalized = normalize(hist)

    # Count higest point
    max_value = 0.0
    index_of_max = 0

    for i in range(0, normalized.shape[0]):
        if normalized[i] > max_value:
            max_value = normalized[i]
            index_of_max = i

    if normalized.shape[0] == 1 or normalized.shape[0] == 0:
        return 0.0

    # Cut from side if max is on one side
    if index_of_max < 10:
        for i in range(160, 180):
            normalized[i] = 0
    elif index_of_max > 170:
        for i in range(0, 20):
            normalized[i] = 0

    # Calculate dispersion
    return calc_standard_deviation(normalized)


def load_image_pixel_location_data(gray_image):
    location_data_list = []

    for y in range(0, gray_image.shape[0]):
        for x in range(0, gray_image.shape[1]):
            location_data_list.append((x, y, gray_image[y, x]))

    return location_data_list


def get_original_image_data(location_data, original_image):
    image_data = []

    for i in range(0, len(location_data)):
        original_pixel = original_image[location_data[i][1], location_data[i][0]]
        image_data.append(original_pixel)

    original_shape =  len(image_data)
    if original_shape % 6 != 0:
        add_to = 6 - (original_shape % 6)
        for i in range(0, add_to):
            image_data.append([0, 0, 0])

    as_image = np.array(image_data)
    return np.reshape(as_image, (-1, 2, 3)).astype(np.uint8)


def retrieve_darkest(sorted_location_data_list, prosent):
    darkest = []

    for i in range(0, int(prosent * len(sorted_location_data_list))):
        darkest.append(sorted_location_data_list[i])

    return darkest


def retrieve_ligthest(sorted_location_data_list, prosent):
    ligthest = []

    for i in range(len(sorted_location_data_list) - 1, len(sorted_location_data_list) - 1 - int(prosent * len(sorted_location_data_list)), -1):
        ligthest.append(sorted_location_data_list[i])

    return ligthest


def average_peak_value_of_largest(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [180], [0,180])

    hist = normalize(hist)

    if hist.shape[0] == 0 or hist.shape[0] == 1:
        return 0.0

    largest_found = largest(calculate_peak_value(hist), 0.2)

    if largest_found.shape[0] == 0:
        return 0.0

    return np.average(largest_found)


def sum_of_areas_with_high_rise_rate(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [180], [0,180])

    hist = normalize(hist)

    continuous_distribution = calculate_continuous_distribution(hist)
    continuous_distribution_derivate = calculate_derivates(continuous_distribution)

    thresh = np.average(largest(continuous_distribution_derivate, 0.25))

    max_cap = 2
    areas = []
    status = 'outside'

    current_area_size = 0
    current_cap = 0

    for i in range(0, continuous_distribution_derivate.shape[0]):
        # Calculate derivate
        derivate = continuous_distribution_derivate[i]

        # Area start
        if derivate >= thresh and status == 'outside':
            status = 'inside'
            current_area_size = 1
            current_cap = 0

        # Check if just small cap
        elif derivate < thresh and status == 'inside' and current_cap < max_cap:
            current_cap += 1
            current_area_size += 1

        # Area end
        elif derivate < thresh and status == 'inside':
            # Remove ending cap
            current_area_size -= current_cap

            areas.append(current_area_size)

            status = 'outside'
            current_cap = 0
            current_area_size = 0

        # Area continues
        elif status == 'inside':
            current_cap = 0
            current_area_size += 1

    return np.sum(np.array(areas))



def get_input_vector(color_image):
    # Calculate darkest and ligthest pixels in image
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    location_data_list = load_image_pixel_location_data(gray_image)
    location_data_list.sort(key = itemgetter(2))

    ligthest = retrieve_ligthest(location_data_list, 0.2)
    darkest = retrieve_darkest(location_data_list, 0.2)

    image_data_ligth = get_original_image_data(ligthest, color_image)
    image_data_darkest = get_original_image_data(darkest, color_image)

    # Calculate peak value
    prediction1 = np.average( np.array( [average_peak_value_of_largest(image_data_ligth), average_peak_value_of_largest(image_data_darkest)] ) )

    # Calculate large areas
    prediction2 = np.max( np.array( [sum_of_areas_with_high_rise_rate(image_data_ligth), sum_of_areas_with_high_rise_rate(image_data_darkest)] ) )

    # Calculate dispersion
    prediction3 = np.max( np.array([count_dispersion(image_data_ligth), count_dispersion(image_data_darkest)]) )

    return np.array([prediction1, prediction2, prediction3]).astype(np.float32)


class CrossProcessed(SVMFilter):

    name = 'cross_processed'
    speed = 5

    def __init__(self, threshold=0.5, invert_threshold=False, svm_file=None):
        """Initializes a Cross Processed image filter

        :param threshold: threshold at which the given prediction is changed
                          from negative to positive
        :type threshold: float
        :param invert_threshold: whether the result should be greater than
                                 the given threshold (default) or lower
                                 for an image to be considered positive
        :type invert_threshold: bool
        :param svm_file: path to a file to load an SVM model from, overrides
                         the default SVM model
        :type svm_file: str
        """
        if svm_file is None:
            super(CrossProcessed, self).__init__(threshold, invert_threshold,
                                      get_data('svm/cross_processed.yml'))
        else:
            super(CrossProcessed, self).__init__(threshold, invert_threshold, svm_file)


    def predict(self, image_path, return_boolean=True, ROI=None):
        """Predict if a given image is a Cross Processed image

        :param image_path: path to the image
        :type image_path: str
        :param return_boolean: whether to return the result as a
                               float between 0 and 1 or as a boolean
                               (threshold is given to the class)
        :type return_boolean: bool
        :param ROI: possible region of interest as a 4-tuple
                    (x0, y0, width, height), None if not needed
        :returns: the prediction as a bool or float depending on the
                  return_boolean parameter
        """
        vector = get_input_vector(read_color_image(image_path, ROI))
        prediction = scaled_prediction(self.svm.predict(vector))

        if return_boolean:
            return self.boolean_result(prediction)
        return prediction


    def train(self, images, labels, save_path=None):
        """Retrain the filter with new training images. The new
        model needs to be saved with the save function for later
        use.

        :param images: list of image paths to training images
        :type images: list
        :param labels: list of labels associated with the images,
                       0 for negative and 1 for positive
        :type labels: list
        :param save_path: possible filepath to save the resulting
                          model to, None if not needed
        :type save_path: str
        """
        super(CrossProcessed, self).train(images, labels, save_path,
                               cv2.imread, get_input_vector)
