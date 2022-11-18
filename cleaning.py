import os
import numpy as np
import mne
from mne.preprocessing import annotate_muscle_zscore
from mne.preprocessing import ICA

from paths import paths
import setup
import load

save_path = paths().save_path()
plot_path = paths().plots_path()
exp_info = setup.exp_info()

for subject_code in exp_info.subjects_ids:

    # Load data
    subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)
    meg_data = subject.load_preproc_meg(preload=True)

    # Visual annotation
    # fig = meg_data.plot(start=0, duration=6)
    # fig.fake_keypress('a')

    # Annotate cardiac artifacts
    ecg_events = mne.preprocessing.find_ecg_events(meg_data)[0]

    n_beats = len(ecg_events)
    onset = ecg_events[:, 0] / meg_data.info['sfreq'] - 0.25
    duration = np.repeat(0.5, n_beats)
    description = ['beats'] * n_beats
    orig_time = meg_data.info['meas_date']
    annotations_heart = mne.Annotations(onset, duration, description, orig_time)

    # Make epochs
    ecg_epochs = mne.preprocessing.create_ecg_epochs(meg_data)
    ecg_epochs.plot_image(combine='mean')

    # Average evoked
    ecg_evoked = ecg_epochs.average()
    ecg_evoked.apply_baseline((None, -0.2))
    ecg_evoked.plot_joint()

    #--------- Muscle artifacts ---------#
    threshold_muscle = 10
    annotations_muscle, scores_muscle = annotate_muscle_zscore(meg_data, ch_type="mag", threshold=threshold_muscle,
                                                               min_length_good=0.5, filter_freq=[110, 140])

    # Include annotations in data
    meg_data.set_annotations(annotations_muscle)

    path_file_results = os.path.join(paths().save_path(), 'training_rawann-1.fif')
    meg_data.save(path_file_results, overwrite=True)

    # Downsamples for ica
    meg_downsampled = meg_data.copy().pick_types(meg=True)
    meg_downsampled.resample(200)
    meg_downsampled.filter(1, 40)

    # Define iCA
    ica = ICA(method='fastica', random_state=97, n_components=64, verbose=True)

    # Apply ICA
    ica.fit(meg_downsampled, verbose=True)

    # Plot sources and components
    ica.plot_sources(meg_downsampled, title='ICA')
    ica.plot_components()

    # Exclude bad components from data
    ica.exclude = [0, 1, 14, 24]  # Correct numbers
    meg_ica = meg_data.copy()
    ica.apply(meg_ica)

    # Save ICA data
    path_outfile = paths().save_path() + 'ICA/' + subject.subject_id + f'/Subject_{subject.subject_id}_ICA.fif'
    meg_ica.save(path_outfile, overwrite=True)

    # Plot to check
    chs = ['MLF14-4123', 'MLF13-4123', 'MLF12-4123', 'MLF11-4123', 'MRF11-4123', 'MRF12-4123', 'MRF13-4123',
           'MRF14-4123', 'MZF01-4123']
    chan_idxs = [meg_data.ch_names.index(ch) for ch in chs]

    meg_data.plot(order=chan_idxs, duration=5)
    meg_ica.plot(order=chan_idxs, duration=5)

