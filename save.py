import mne
import os
import pickle
import pandas as pd
import pathlib

from paths import paths


def preprocesed(raw, subject, bh_data, fixations, saccades):
    #---------------- Save preprocesed data ----------------#
    print('Saving preprocessed data')
    # Path
    preproc_data_path = paths().preproc_path()
    preproc_save_path = preproc_data_path + subject.subject_id + '/'
    os.makedirs(preproc_save_path, exist_ok=True)

    # Add data to subject class and save
    subject.bh_data = bh_data
    subject.fixations = fixations
    subject.saccades = saccades

    f = open(preproc_save_path + 'Subject_data.pkl', 'wb')
    pickle.dump(subject, f)
    f.close()

    # Save fixations
    fixations.to_csv(preproc_save_path + 'fixations.csv')

    # Save MEG
    preproc_meg_data_fname = f'Subject_{subject.subject_id}_meg.fif'
    raw.save(preproc_save_path + preproc_meg_data_fname, overwrite=True)

    # Save events
    evt, evt_id = mne.events_from_annotations(raw)
    preproc_evt_data_fname = f'Subject_{subject.subject_id}_eve.fif'
    mne.write_events(preproc_save_path + preproc_evt_data_fname, evt, overwrite=True)

    # Save events mapping
    evt_df = pd.DataFrame([evt_id])
    preproc_evt_map_fname = f'Subject_{subject.subject_id}_eve_map.csv'
    evt_df.to_csv(preproc_save_path + preproc_evt_map_fname)

    print(f'Preprocessed data saved to {preproc_save_path}')


def preproc_config(subject, config):
    '''
    Configuration setup for preprocessing run.
    '''

    print('\nSaving preprocessing configuration')

    # Configuration path
    file_path = pathlib.Path(os.path.join(subject.config_path, subject.subject_id, f'{subject.subject_id}_config.pkl'))

    f = open(file_path, 'wb')
    pickle.dump(config, f)
    f.close()

