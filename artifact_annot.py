import os
from mne.preprocessing import annotate_muscle_zscore
from mne.preprocessing import ICA

from paths import paths
import setup
import load

ica_path = paths().ica_path()
plot_path = paths().plots_path()
exp_info = setup.exp_info()

for subject_code in exp_info.subjects_ids:

    # Load data
    subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)
    meg_data = subject.load_preproc_meg(preload=True)

    # #--------- Visual annotation ---------#
    # fig = meg_data.plot(start=0, duration=6)
    # fig.fake_keypress('a')

    # #--------- Cardiac artifacts ---------#
    # ecg_events = mne.preprocessing.find_ecg_events(meg_data)[0]
    #
    # n_beats = len(ecg_events)
    # onset = ecg_events[:, 0] / meg_data.info['sfreq'] - 0.25
    # duration = np.repeat(0.5, n_beats)
    # description = ['beats'] * n_beats
    # orig_time = meg_data.info['meas_date']
    # annotations_heart = mne.Annotations(onset, duration, description, orig_time)
    #
    # # Make epochs
    # ecg_epochs = mne.preprocessing.create_ecg_epochs(meg_data)
    # ecg_epochs.plot_image(combine='mean')
    #
    # # Average evoked
    # ecg_evoked = ecg_epochs.average()
    # ecg_evoked.apply_baseline((None, -0.2))
    # ecg_evoked.plot_joint()

    #--------- Muscle artifacts ---------#
    threshold_muscle = 5
    annotations_muscle, scores_muscle = annotate_muscle_zscore(meg_data, ch_type="mag", threshold=threshold_muscle,
                                                               min_length_good=0.2, filter_freq=[110, 140])

    # Include annotations in data
    meg_data.set_annotations(annotations_muscle)