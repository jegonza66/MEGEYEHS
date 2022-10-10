from paths import paths
import setup
import os
import pathlib
import pickle


def config(config_path):
    """
    Load the run configuration and setup information.

    Attributes
    ----------
    exp_info: class
        The experiment information class.

    Returns
    -------
    config: class
        Class containgn the run configuration and setup information.
    """

    try:
        # Load
        filepath = config_path + '/config.pkl'
        f = open(filepath, 'rb')
        config = pickle.load(f)
        f.close()

        # Set save config as false
        config.update_config = False

    except:
        # Create if no previous configuration file
        config = setup.config()

    return config


def preproc_subject(exp_info, subject_code):
    """
    Preprocessed subject class

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
        raise ValueError(f'Preprocessed data for subject {subject_id} not found')







