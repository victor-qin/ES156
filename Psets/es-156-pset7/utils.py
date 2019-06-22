import matplotlib.pyplot as plt
from pydub import AudioSegment
import pyaudio
import numpy as np
import os
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import (generate_binary_structure,
                                      iterate_structure, binary_erosion)


FORMAT = pyaudio.paInt16
CHANNELS = 2
SAMPLE_RATE = 44100
CHUNK_SIZE = 8192
DEFAULT_AMP_MIN = 3.5
PEAK_NEIGHBORHOOD_SIZE = 20


'''
get_2D_peaks() taken/adapted from: https://github.com/worldveil/dejavu
'''
def get_2D_peaks(arr2D, amp_min=DEFAULT_AMP_MIN):
    # http://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.morphology.iterate_structure.html#scipy.ndimage.morphology.iterate_structure
    struct = generate_binary_structure(2, 1)
    neighborhood = iterate_structure(struct, PEAK_NEIGHBORHOOD_SIZE)

    # find local maxima using our fliter shape
    local_max = maximum_filter(arr2D, footprint=neighborhood) == arr2D
    background = (arr2D == 0)
    eroded_background = binary_erosion(background, structure=neighborhood,
                                       border_value=1)

    # Boolean mask of arr2D with True at peaks
    detected_peaks = local_max ^ eroded_background

    # extract peaks
    amps = arr2D[detected_peaks]
    j, i = np.where(detected_peaks)

    # filter peaks
    amps = amps.flatten()
    peaks = zip(i, j, amps)
    peaks_filtered = [x for x in peaks if x[2] > amp_min]  # freq, time, amp

    # get indices for frequency and time
    frequency_idx = [x[1] for x in peaks_filtered]
    time_idx = [x[0] for x in peaks_filtered]

    return frequency_idx, time_idx, zip(frequency_idx, time_idx)


# For (potential) live recording
def getRecording(seconds):
	data = [[] for i in range(CHANNELS)]

	# pyaudio stream
	stream = pyaudio.PyAudio().open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
        )

	# Record and process audio with multiple channels
	for i in range(0, int(SAMPLE_RATE/CHUNK_SIZE * seconds)):
		d = stream.read(CHUNK_SIZE)
		nums = np.fromstring(d, np.int16)
		for c in range(CHANNELS):
			data[c].extend(nums[c::CHANNELS])

	# Stop recording
	stream.stop_stream()
	stream.close()
	stream = None
	recorded = True
	return data
