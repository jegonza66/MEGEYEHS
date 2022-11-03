import matplotlib.pyplot as plt
import mne
import os
import pickle
import pandas as pd

from paths import paths


def preprocesed_data(raw, subject, config):
    """
    Save preprocesed data
    :param raw:
    :param subject:
    :param bh_data:
    :param fixations:
    :param saccades:
    """

    print('Saving preprocessed data')
    # Path
    preproc_data_path = paths().preproc_path()
    preproc_save_path = preproc_data_path + subject.subject_id + '/'
    os.makedirs(preproc_save_path, exist_ok=True)

    f = open(preproc_save_path + 'Subject_data.pkl', 'wb')
    pickle.dump(subject, f)
    f.close()

    # Save fixations ans saccades
    subject.fixations.to_csv(preproc_save_path + 'fixations.csv')
    subject.saccades.to_csv(preproc_save_path + 'saccades.csv')

    # Save MEG
    preproc_meg_data_fname = f'Subject_{subject.subject_id}_meg.fif'
    raw.save(preproc_save_path + preproc_meg_data_fname, overwrite=True)

    # Save events
    evt, evt_id = mne.events_from_annotations(raw, verbose=False)
    preproc_evt_data_fname = f'Subject_{subject.subject_id}_eve.fif'
    mne.write_events(preproc_save_path + preproc_evt_data_fname, evt, overwrite=True)

    # Save events mapping
    evt_df = pd.DataFrame([evt_id])
    preproc_evt_map_fname = f'Subject_{subject.subject_id}_eve_map.csv'
    evt_df.to_csv(preproc_save_path + preproc_evt_map_fname)

    # Save configuration
    config.update_config = False
    config_path = paths().config_path()
    var(config, path=config_path, fname='config.pkl')

    print(f'Preprocessed data saved to {preproc_save_path}')


def var(var, path, fname):
    """
    Save variable var with given filename to given path.

    Parameters
    ----------
    var: any
        Variable to save
    path: str
        Path to save directory
    fname: str
        Filename of file to save
    """

    # Make dir
    os.makedirs(path, exist_ok=True)

    # Save
    file_path = path + fname
    f = open(file_path, 'wb')
    pickle.dump(var, f)
    f.close()


def fig(fig, path, fname):
    """
    Save figure fig with given filename to given path.

    Parameters
    ----------
    fig: figure
        Instance of figure to save
    path: str
        Path to save directory
    fname: str
        Filename of file to save
    """

    # Make dir
    os.makedirs(path, exist_ok=True)

    # Create svg directory
    svg_path = path + 'svg/'
    os.makedirs(svg_path, exist_ok=True)

    # Save
    fig.savefig(path + fname)
    fig.savefig(svg_path + fname)

