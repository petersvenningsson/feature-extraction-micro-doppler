###########
# IMPORTS #
###########
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import minimum_filter
from scipy.stats import trim_mean
from scipy import interpolate


def clean_spectrogram(spectrogram, doppler_bins):

    n_dbins = spectrogram.shape[0]
    for column in spectrogram.T:
        indices_to_interpolate = set(range(17,31))
        indices_to_keep = set(range(n_dbins)) - indices_to_interpolate

        indices_to_interpolate = np.sort(list(indices_to_interpolate))
        indices_to_keep = np.sort(list(indices_to_keep))

        interpolated_values = interpolate.griddata(indices_to_keep, column[indices_to_keep], indices_to_interpolate, method='linear')
        column[indices_to_interpolate] = interpolated_values

    return spectrogram, doppler_bins


def get_centriod(spectrogram, doppler_bins):
    
    centriods = []
    for column in spectrogram.T:
        normalization_constant = np.sum(column)
        _centriod = [ doppler_bins[i]*column[i] for i in range(column.size) ]
        centriod = np.sum(_centriod) / normalization_constant
        centriods.append(centriod)
    return centriods


def get_bandwidth(spectrogram, doppler_bins):
    centriods = get_centriod(spectrogram, doppler_bins)
    
    bandwidths = []
    for j, column in enumerate(spectrogram.T):
        normalization_constant = np.sum(column)
        _bandwidth = [ ( doppler_bins[i] - centriods[j] )**2 * column[i] for i in range(column.size) ]
        bandwidth = np.sqrt( np.sum(_bandwidth)/normalization_constant )
        bandwidths.append( bandwidth )

    return bandwidths


def get_span(spectrogram, doppler_bins, threshold):

    mean = np.mean(spectrogram)
    std = np.std(spectrogram, ddof = 1)
    contour_threshold = mean + threshold * std

    spectrogram, doppler_bins = clean_spectrogram(spectrogram, doppler_bins)

    # Calculate std bandwidth
    upper_contour = []
    lower_contour = []
    for column in spectrogram.T:
        if any(np.argwhere(column > contour_threshold)):
            upper_contour.append(np.max(np.argwhere(column > contour_threshold)))
            lower_contour.append(np.min(np.argwhere(column > contour_threshold)))
        else:
            upper_contour.append(np.argmax(column))
            lower_contour.append(np.argmin(column))
    upper_contour = np.array(upper_contour)
    lower_contour = np.array(lower_contour)
    countour_range = np.array(range(len( upper_contour )))
    return upper_contour, lower_contour, countour_range


def extract_silhouette_size(spectrogram, doppler_bins):

    mean = np.mean(spectrogram)
    std = np.std(spectrogram, ddof = 1)
    threshold = mean - 0.05*std
    
    spectrogram, doppler_bins = clean_spectrogram(spectrogram, doppler_bins)

    silhouette = spectrogram > threshold
    silhouette_size = np.sum(silhouette)/spectrogram.size

    return silhouette_size


def extract_peak_spread(spectrogram, doppler_bins, min_peak_height = 0):

    mean = np.mean(spectrogram)
    std = np.std(spectrogram, ddof = 1)
    threshold = mean - 0.05*std

    spectrogram, doppler_bins = clean_spectrogram(spectrogram, doppler_bins)

    # Min filter spectrogram in doppler to create clearer countour.
    filtered_spectrogram = minimum_filter(spectrogram, size = (8,2))
    contour = []
    for column in filtered_spectrogram.T:
        if any(np.argwhere(column > threshold)):
            contour.append(np.max(np.argwhere(column > threshold)))
        else:
            contour.append(np.argmax(column))
    contour = np.array(contour)

    countour_range = np.array(range(len( contour )))
    peaks, _ = find_peaks(contour, prominence=7, distance = 15, width = 3, height = min_peak_height)

    centroid = np.argmax(spectrogram, axis = 0)
    mean_centriod = trim_mean(centroid, proportiontocut= 0.2)
    if peaks.size != 0:
        max_nominal_velocity = doppler_bins[np.max(contour[peaks])] - doppler_bins[int(mean_centriod)]

        min_nominal_velocity = doppler_bins[np.min(contour[peaks])] - doppler_bins[int(mean_centriod)]

        peak_spread = max_nominal_velocity - min_nominal_velocity
    else:
        peak_spread = 0

    return peak_spread


def extract_peak_height(spectrogram, doppler_bins, min_peak_height = 0):

    mean = np.mean(spectrogram)
    std = np.std(spectrogram, ddof = 1)
    threshold = mean - 0.05*std
    
    spectrogram, doppler_bins = clean_spectrogram(spectrogram, doppler_bins)

    # Min filter spectrogram in doppler to create clearer countour.
    filtered_spectrogram = minimum_filter(spectrogram, size = (8,2))
    contour = []
    for column in filtered_spectrogram.T:
        if any(np.argwhere(column > threshold)):
            contour.append(np.max(np.argwhere(column > threshold)))
        else:
            contour.append(np.argmax(column))
    contour = np.array(contour)

    countour_range = np.array(range(len( contour )))
    peaks, _ = find_peaks(contour, prominence=7, distance = 15, width = 3, height = min_peak_height)

    centroid = np.argmax(spectrogram, axis = 0)
    mean_centriod = trim_mean(centroid, proportiontocut= 0.2)
    mean_centroid_doppler = doppler_bins[int(mean_centriod)]

    if peaks.size != 0:
        peak_doppler_values = doppler_bins[contour[peaks]]
        mean_peak_doppler = np.mean(peak_doppler_values - mean_centroid_doppler)
    else:
        mean_peak_doppler = doppler_bins[np.max(contour)]

    return mean_peak_doppler


def extract_mean_span(spectrogram, doppler_bins):

    upper_contour, lower_contour, countour_range = get_span(spectrogram, doppler_bins, threshold=0.3)
    mean_span = np.mean(doppler_bins[upper_contour] - doppler_bins[lower_contour])

    return mean_span


def extract_std_span(spectrogram, doppler_bins):

    upper_contour, lower_contour, countour_range = get_span(spectrogram, doppler_bins, threshold=0.3)
    std_span = np.std(doppler_bins[upper_contour] - doppler_bins[lower_contour], ddof = 1)

    return std_span


def extract_mean_centriod(spectrogram, doppler_bins):

    spectrogram, doppler_bins = clean_spectrogram(spectrogram, doppler_bins)
    centroid = get_centriod(spectrogram, doppler_bins)
    mean_centroid = np.mean( centroid )

    return mean_centroid


def extract_std_centriod(spectrogram, doppler_bins, render = False, render_time = None, idstr = None):

    spectrogram, doppler_bins = clean_spectrogram(spectrogram, doppler_bins)
    centroid = get_centriod(spectrogram, doppler_bins)
    std_centroid = np.std( centroid, ddof = 1 )

    return std_centroid


def extract_mean_bandwidth(spectrogram, doppler_bins, render = False, render_time = None, idstr = None):
    """ Extracts the mean time between peaks in the spectrogram.
    """
    spectrogram, doppler_bins = clean_spectrogram(spectrogram, doppler_bins)
    bandwidth = get_bandwidth(spectrogram, doppler_bins)
    mean_bandwidth = np.mean( bandwidth )

    return mean_bandwidth


def extract_std_bandwidth(spectrogram, doppler_bins, render = False, render_time = None, idstr = None):
    """ Extracts the mean time between peaks in the spectrogram.
    """
    spectrogram, doppler_bins = clean_spectrogram(spectrogram, doppler_bins)
    bandwidth = get_bandwidth(spectrogram, doppler_bins)
    std_bandwidth = np.std( bandwidth, ddof = 1)

    return std_bandwidth
