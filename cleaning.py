import os
import sys
import numpy as np
from mne.preprocessing import annotate_muscle_zscore
import mne

from paths import paths
import setup
import load

save_path = paths().save_path()
plot_path = paths().plots_path()
exp_info = setup.exp_info()

for subject_code in exp_info.subjects_ids:

    # Load data
    subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)
    meg_data = subject.load_preproc_meg()

    eog_events = mne.preprocessing.find_eog_events(meg_data, ch_name='EOG001')

    n_blinks = len(eog_events)
    onset = eog_events[:, 0] / meg_data.info['sfreq'] - 0.25
    duration = np.repeat(0.5, n_blinks)
    description = ['blink'] * n_blinks
    orig_time = meg_data.info['meas_date']
    annotations_blink = mne.Annotations(onset, duration, description, orig_time)



    # Muscle artifacts
    threshold_muscle = 10
    annotations_muscle, scores_muscle = annotate_muscle_zscore(meg_data, ch_type="mag", threshold=threshold_muscle,
                                                               min_length_good=0.2, filter_freq=[110, 140])

    # Include annotations in data
    meg_data.set_annotations(annotations_blink + annotations_muscle)

    meg_data.plot(start=50);

    # Set the channel type as 'eog'
    meg_data.set_channel_types({'EOG001': 'eog'})
    meg_data.set_channel_types({'EOG002': 'eog'})

    eog_picks = mne.pick_types(meg_data.info, meg=False, eog=True)

    scl = dict(eog=500e-6)
    meg_data.plot(order=eog_picks, scalings=scl, start=50)

    path_file_results = os.path.join(paths().save_path(), 'training_rawann-1.fif')
    path_file_results
    meg_data.save(path_file_results, overwrite=True)