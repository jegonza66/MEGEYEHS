import setup
import functions_preproc
import load
import copy
import numpy as np
import os
from paths import paths
import save
import mne
import pandas as pd
import plot_preproc

# Load experiment info
exp_info = setup.exp_info()
config = setup.config()

# Run
for subject_code in exp_info.subjects_ids[1:]:

    # ---------------- Load data ----------------#
    # Define subject
    subject_ica = load.ica_subject(exp_info=exp_info, subject_code=subject_code)
    subject_preproc = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)

    # Load MEG data
    ica_data = load.ica_data(subject=subject_ica, preload=True)
    preproc_data = subject_preproc.load_preproc_meg_data(preload=True)

    # Get fixations from subject object
    fixations = subject_ica.fixations
    # Detect fixations to MS items
    raw, subject, ms_items_pos = functions_preproc.ms_items_fixations(fixations=copy.copy(subject_ica.fixations), subject=copy.copy(subject_ica),
                                                                      raw=ica_data.copy(), distance_threshold=70)
    print('Getting et data from meg...')
    et_channels_meg = raw.get_data(picks=['ET_gaze_x', 'ET_gaze_y', 'ET_pupils'])

    print('Plotting scanpaths and trials gaze screens')
    for trial_idx in range(len(subject.bh_data)):
        print(f'\rTrial {trial_idx + 1}', end='')

        plot_preproc.ms_scanpath(raw=raw, subject=subject, ms_items_pos=ms_items_pos, et_channels_meg=et_channels_meg, trial_idx=trial_idx)
    print('\n')

    # Overwrite subjects fixations
    subject_ica.fixations = subject.fixations
    subject_preproc.fixations = subject.fixations

    # Get new fixations Dataframe
    fixations_new = subject.fixations

    # Get dissimilarities in id column
    dif_df = ~(fixations['id'] == fixations_new['id'])
    dif_id_old = fixations['id'][dif_df].tolist()
    dif_id_new = fixations_new['id'][dif_df].tolist()

    for id_old, id_new in zip(dif_id_old, dif_id_new):
        description_idx_ica = np.where(ica_data.annotations.description == id_old)[0][0]
        ica_data.annotations.description[description_idx_ica] = id_new

        description_idx_preproc = np.where(preproc_data.annotations.description == id_old)[0][0]
        preproc_data.annotations.description[description_idx_preproc] = id_new

    # Save
    # Preproc subject
    preproc_save_path = paths().preproc_path() + subject.subject_id + '/'
    save.var(var=subject_preproc, path=preproc_save_path, fname='Subject_data.pkl')
    # ICA subject
    save_path_ica = paths().ica_path() + subject.subject_id + '/'
    save.var(var=subject_ica, path=save_path_ica, fname='Subject_data.pkl')

    # Save MEG
    preproc_meg_data_fname = f'Subject_{subject.subject_id}_meg.fif'
    preproc_data.save(preproc_save_path + preproc_meg_data_fname, overwrite=True)

    path_file = os.path.join(save_path_ica, f'Subject_{subject.subject_id}_ICA.fif')
    ica_data.save(path_file, overwrite=True)

    # Save fixations and saccades
    fixations_new.to_csv(preproc_save_path + 'fixations.csv')

    # Save events
    evt, evt_id = mne.events_from_annotations(preproc_data, verbose=False)
    preproc_evt_data_fname = f'Subject_{subject.subject_id}_eve.fif'
    mne.write_events(preproc_save_path + preproc_evt_data_fname, evt, overwrite=True)

    # Save events mapping
    evt_df = pd.DataFrame([evt_id])
    preproc_evt_map_fname = f'Subject_{subject.subject_id}_eve_map.csv'
    evt_df.to_csv(preproc_save_path + preproc_evt_map_fname)


