from paths import paths
import setup
import os
import pathlib
import pickle
import mne
import functions_general
import glob
import pandas as pd


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
    except:
        print(f'Directory: {os.listdir(pathlib.Path(os.path.join(preproc_path, subject_id)))}')
        raise ValueError(f'Preprocessed data for subject {subject_id} not found in {file_path}')

    return preproc_subject


def ica_subject(exp_info, subject_code):
    """
    Load ica subject object.

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
    ica_subject: class
        The ica subject class
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
    ica_path = paths().ica_path()
    file_path = pathlib.Path(os.path.join(ica_path, subject_id, f'Subject_data.pkl'))
    try:
        f = open(file_path, 'rb')
        ica_subject = pickle.load(f)
        f.close()
    except:
        try:
            # Retry with pandas
            ica_subject = pd.read_pickle(file_path)
        except:
            print(f'Directory: {os.listdir(pathlib.Path(os.path.join(ica_path, subject_id)))}')
            raise ValueError(f'ICA data for subject {subject_id} not found in {file_path}')

    return ica_subject


def downsampled_data(subject, sfreq, band_id=None, method='iir', preload=False, save_data=False):

    ds_path = paths().ds_path_ica() + f'Band_{band_id}/{subject.subject_id}/'

    ds_meg_data_fname = f'Subject_{subject.subject_id}_method_{method}_meg.fif'

    # Try to load filtered data
    try:
        print(f'Loading downsampled data data in band {band_id} for subject {subject.subject_id}')
        # Load data
        ds_data = mne.io.read_raw_fif(ds_path + ds_meg_data_fname, preload=preload)

    except:
        print(f'No previous downsampled data found for subject {subject.subject_id} in band {band_id}.\n'
              f'Downsampleing data...')

        ds_data = filtered_data(subject=subject, band_id=band_id, save_data=save_data,
                                 method=method)

        print(f'Downsampling data to {sfreq} Hz')
        ds_data.resample(sfreq)

        if save_data:
            print('Saving filtered data')
            # Save MEG
            os.makedirs(ds_path, exist_ok=True)
            ds_data.save(ds_path + ds_meg_data_fname, overwrite=True)

    return ds_data


def filtered_data(subject, band_id, method='iir', use_ica_data=True, preload=False, save_data=False):

    if use_ica_data:
        filtered_path = paths().filtered_path_ica() + f'{band_id}/{subject.subject_id}/'
    else:
        filtered_path = paths().filtered_path_raw() + f'{band_id}/{subject.subject_id}/'

    filtered_meg_data_fname = f'Subject_{subject.subject_id}_method_{method}_meg.fif'

    # Try to load filtered data
    try:
        print(f'Loading filtered data in band {band_id} for subject {subject.subject_id}')
        # Load data
        filtered_data = mne.io.read_raw_fif(filtered_path + filtered_meg_data_fname, preload=preload)

    except:
        print(f'No previous filtered data found for subject {subject.subject_id} in band {band_id}.\n'
              f'Filtering data...')

        if use_ica_data:
            meg_data = ica_data(subject=subject, preload=True)
        else:
            meg_data = subject.load_preproc_meg_data(preload=True)

        l_freq, h_freq = functions_general.get_freq_band(band_id)
        if method:
            filtered_data = meg_data.filter(l_freq=l_freq, h_freq=h_freq, method=method)
        else:
            filtered_data = meg_data.filter(l_freq=l_freq, h_freq=h_freq)

        if save_data:
            print('Saving filtered data')
            # Save MEG
            os.makedirs(filtered_path, exist_ok=True)
            filtered_data.save(filtered_path + filtered_meg_data_fname, overwrite=True)

    return filtered_data


def ica_data(subject, preload=False):

    path = paths().ica_path() + f'{subject.subject_id}/'

    # Try to load ica data
    try:
        print(f'Loading ica data for subject {subject.subject_id}')
        # Load data
        ica_data = mne.io.read_raw_fif(path + f'Subject_{subject.subject_id}_ICA.fif', preload=preload)

    except:
        raise ValueError(f'No previous ica data found for subject {subject.subject_id}')

    return ica_data


def meg(subject, meg_params, save_data=False):

    if meg_params['data_type'] == 'ICA':
        if 'downsample' in meg_params.keys() and meg_params['downsample']:
            meg_data = downsampled_data(subject=subject, sfreq=meg_params.get('downsample'), band_id=meg_params.get('band_id', None),
                                       method=meg_params.get('filter_method', 'iir'), save_data=save_data)
        elif meg_params.get('band_id') and meg_params['filter_sensors']:
            meg_data = filtered_data(subject=subject, band_id=meg_params['band_id'], save_data=save_data,
                                          method=meg_params['filter_method'])
        else:
            meg_data = ica_data(subject=subject)
    elif meg_params['data_type'] == 'RAW':
        if meg_params.get('band_id') and meg_params.get('filter_sensors'):
            meg_data = filtered_data(subject=subject, band_id=meg_params['band_id'], use_ica_data=False,
                                          save_data=save_data, method=meg_params['filter_method'])
        else:
            meg_data = subject.load_preproc_meg_data()

    return meg_data


def time_frequency_range(file_path, l_freq, h_freq):

    # MS difference
    matching_files_ms = glob.glob(file_path.replace(f'{l_freq}_{h_freq}', '*'))
    if len(matching_files_ms):
        for file in matching_files_ms:
            l_freq_file = int(file.split('_')[-3])
            h_freq_file = int(file.split('_')[-2])

            # If file contains desired frequencies, Load
            if l_freq_file <= l_freq and h_freq_file >= h_freq:
                time_frequency = mne.time_frequency.read_tfrs(file)[0]

                # Crop to desired frequencies
                time_frequency = time_frequency.crop(fmin=l_freq, fmax=h_freq)
                break
            else:
                raise ValueError('No file found with desired frequency range')
    else:
        raise ValueError('No file found with desired frequency range')

    return time_frequency


def forward_model(sources_path_subject, subject_code, chs_id, source_params):

    if source_params['surf_vol'] == 'volume':
        fname_fwd = sources_path_subject + f"/{subject_code}_volume_ico{source_params['ico']}_{int(source_params['spacing'])}-fwd.fif"
    elif source_params['surf_vol'] == 'surface':
        fname_fwd = sources_path_subject + f"/{subject_code}_surface_ico{source_params['ico']}-fwd.fif"
    elif source_params['surf_vol'] == 'mixed':
        fname_fwd = sources_path_subject + f"/{subject_code}_mixed_ico{source_params['ico']}_{int(source_params['spacing'])}-fwd.fif"
    fwd = mne.read_forward_solution(fname_fwd)

    return fwd


def source_model(sources_path_subject, subject_code, source_params):

    # Get Source space for default subject
    if source_params['surf_vol'] == 'volume':
        fname_src = sources_path_subject + f"/fsaverage_volume_ico{source_params['ico']}_{int(source_params['spacing'])}-src.fif"
    elif source_params['surf_vol'] == 'surface':
        fname_src = sources_path_subject + f"/fsaverage_surface_ico{source_params['ico']}-src.fif"
    elif source_params['surf_vol'] == 'mixed':
        fname_src = sources_path_subject + f"/fsaverage_mixed_ico{source_params['ico']}_{int(source_params['spacing'])}-src.fif"

    src = mne.read_source_spaces(fname_src)

    return src