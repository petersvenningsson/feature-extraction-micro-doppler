from matplotlib import pyplot as plt
from scipy.io import loadmat
import pickle
import doppler_features as df


def render_spectrogram(spectrogram, doppler_axis, time_axis):
    spectrogram, doppler_axis = df.clean_spectrogram(spectrogram, doppler_axis)
    plt.imshow(spectrogram, aspect='auto', cmap='jet', origin='lower', vmin=-40, extent = [time_axis[0], int(time_axis[-1]), doppler_axis[0], doppler_axis[-1]])
    plt.colorbar()
    plt.ylabel("Doppler [Hz]")
    plt.xlabel("Time [s]")
    plt.show()


with open('spectrogram.pickle', 'rb') as f:
    spectrogram, doppler_axis, time_axis = pickle.load(f)
render_spectrogram(spectrogram, doppler_axis, time_axis)

characterization = [ 'mean_bandwidth', 'std_bandwidth', 'mean_centriod', 'std_centriod',
        'peak_height','peak_spread', 'silhouette_size',
        'mean_span', 'std_span'] 
extractors = [ getattr(df, f"extract_{f}") for f in characterization]
features = {feature_name:extractor(spectrogram, doppler_axis) for (feature_name, extractor) in zip(characterization, extractors)}

for feature, value in features.items():
    print(f'Feature {feature}: {value}')