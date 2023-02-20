import numpy as np
from mne.preprocessing import annotate_muscle_zscore
import save
import matplotlib.pyplot as plt
plt.figure()
plt.close('all')
from paths import paths
import setup
import load


preproc_path = paths().preproc_path()
plot_path = paths().plots_path()
exp_info = setup.exp_info()

for subject_code in exp_info.subjects_ids[11:]:

    # --------- Load data ---------#
    subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)
    meg_data = subject.load_preproc_meg()

    # --------- Visual annotation ---------#
    meg_data_visual = meg_data.copy()
    fig = meg_data_visual.pick_types(meg=True).plot(duration=25, n_channels=271, scalings=dict(mag=0.6e-12))
    fig.fake_keypress('a')

    #--------- Muscle artifacts ---------#
    print('Loading MEG data into memory...')
    meg_data.load_data()
    threshold_muscle = 5
    print('Running muscle artifact detection...')
    annotations_muscle, scores_muscle = annotate_muscle_zscore(meg_data, ch_type="mag", threshold=threshold_muscle,
                                                               min_length_good=0.2, filter_freq=[110, 140])

    # Include annotations in data by getting bad annotations, muscle anotations and adding them to the meg_data annotations
    annotations_bad = meg_data_visual.annotations
    annotations_bad.delete(np.where(annotations_bad.description != 'bad')[0])
    meg_data.set_annotations(meg_data.annotations + annotations_muscle + annotations_bad)

    # Plot new PSD from annotated data
    fig = meg_data.plot_psd(picks='mag', show=True)
    fig_path = paths().plots_path() + 'Preprocessing/' + subject.subject_id + '/'
    fig_name = 'Annot_PSD'
    save.fig(fig=fig, path=fig_path, fname=fig_name)

    # Save MEG with new annotations and muscle nans
    preproc_save_path = preproc_path + subject.subject_id + '/'
    preproc_meg_data_fname = f'Subject_{subject.subject_id}_meg.fif'
    meg_data.save(preproc_save_path + preproc_meg_data_fname, overwrite=True)

    # Save new annotations
    meg_data.annotations.save(preproc_save_path + 'clean_annotations.csv', overwrite=True)