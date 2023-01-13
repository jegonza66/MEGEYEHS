from paths import paths
import setup
import os
import pathlib
import pickle
import mne
import functions_general


def config(path, fname):
    """
    Try and load the run configuration and setup information.
    If no previous configuration file was saved, setup config obj.

    Parameters
    ----------
    path: str
        The path to the directory where configuration file is stored.
    fname: str
        The filename for the configuration file.

    Returns
    -------
    config: class
        Class containgn the run configuration and setup information.
    """

    try:
        # Load
        filepath = path + fname
        f = open(filepath, 'rb')
        config = pickle.load(f)
        f.close()

        # Set save config as false
        config.update_config = False

    except:
        # Create if no previous configuration file
        config = setup.config()

    return config


def var(file_path):
    """
    Load variable from specified path

    Parameters
    ----------
    file_path: str
        The path to the file to load.

    Returns
    -------
    var: any
        The loaded variable.
    """
    # Load
    f = open(file_path, 'rb')
    var = pickle.load(f)
    f.close()

    return var


def preproc_subject(exp_info, subject_code):
    """
    Load preprocessed subject object.

    Attributes
    --------
    fixations:
    saccades:
    config?:

    Parameters
    --------
    exp_info: class
       Experiment information class

    Returns
    -------
    preproc_subject: class
        The preprocessed subject class
    """

    # Select 1st subject by default
    if subject_code == None:
        subject_id = exp_info.subjects_ids[0]
    # Select subject by index
    elif type(subject_code) == int:
        subject_id = exp_info.subjects_ids[subject_code]
    # Select subject by id
    elif type(subject_code) == str and (subject_code in exp_info.subjects_ids):
        subject_id = subject_code
    else:
        print('Subject not found.')

    # Preprocessing configuration
    preproc_path = paths().preproc_path()
    file_path = pathlib.Path(os.path.join(preproc_path, subject_id, f'Subject_data.pkl'))
    try:
        f = open(file_path, 'rb')
        preproc_subject = pickle.load(f)
        f.close()
        return preproc_subject

    except:
        print(f'Directory: {os.listdir(pathlib.Path(os.path.join(preproc_path, subject_id)))}')
        raise ValueError(f'Preprocessed data for subject {subject_id} not found in {file_path}')


def filtered_data(subject, band_id, preload=False, save_data=False):

    filtered_path = paths().filtered_path() + f'{band_id}/{subject.subject_id}/'

    # Try to load filtered data
    try:
        print(f'Loading filtered data in band {band_id} for subject {subject.subject_id}')
        file_path = pathlib.Path(os.path.join(filtered_path, f'Subject_{subject.subject_id}_meg.fif'))
        # Load data
        filtered_data = mne.io.read_raw_fif(file_path, preload=preload)

    except:
        print(f'No previous filtered data found for subject {subject.subject_id} in band {band_id}.\n'
              f'Filtering data...')
        preproc_data = subject.load_preproc_meg(preload=True)
        l_freq, h_freq = functions_general.get_freq_band(band_id)
        filtered_data = preproc_data.filter(l_freq=l_freq, h_freq=h_freq)

        if save_data:
            print('Saving filtered data')
            # Save MEG
            os.makedirs(filtered_path, exist_ok=True)
            filtered_meg_data_fname = f'Subject_{subject.subject_id}_meg.fif'
            filtered_data.save(filtered_path + filtered_meg_data_fname, overwrite=True)

    return filtered_data


def ica_data(subject, preload=False):

    filtered_path = paths().ica_path() + f'{subject.subject_id}/'

    # Try to load ica data
    try:
        print(f'Loading ica data for subject {subject.subject_id}')
        file_path = pathlib.Path(os.path.join(filtered_path, f'Subject_{subject.subject_id}_ICA.fif'))
        # Load data
        ica_data = mne.io.read_raw_fif(file_path, preload=preload)

    except:
        raise ValueError(f'No previous ica data found for subject {subject.subject_id}')

    return ica_data
