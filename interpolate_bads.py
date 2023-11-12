import setup
import pandas as pd
import mne
import load
from paths import paths

# Load experiment info
exp_info = setup.exp_info()

# Digitalization data path
dig_path = paths().opt_path()

# Preprocessed data save path
preproc_data_path = paths().preproc_path()


for subject_code in exp_info.subjects_ids:

    # Load subject and meg preprocessed data
    subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)
    meg_data = subject.load_preproc_meg_data()

    # Load digitalization file
    dig_path_subject = dig_path + subject.subject_id
    dig_filepath = dig_path_subject + '/Model_Mesh_5m_headers.pos'
    pos = pd.read_table(dig_filepath, index_col=0)

    # Get fiducials from dig
    nasion = pos.loc[pos.index == 'nasion ']
    lpa = pos.loc[pos.index == 'left ']
    rpa = pos.loc[pos.index == 'right ']

    # Get head points
    pos.drop(['nasion ', 'left ', 'right '], inplace=True)
    pos_array = pos.to_numpy()

    # Make montage
    dig_montage = mne.channels.make_dig_montage(nasion=nasion.values.ravel(), lpa=lpa.values.ravel(),
                                                rpa=rpa.values.ravel(), hsp=pos_array, coord_frame='unknown')

    # Make info object
    meg_data.info.set_montage(montage=dig_montage)

    # Load data to run interpolation
    meg_data.load_data()

    # Interpolate bad channels
    meg_data.interpolate_bads()

    # Save as preprocessed
    preproc_save_path = preproc_data_path + subject.subject_id + '/'
    preproc_meg_data_fname = f'Subject_{subject.subject_id}_meg.fif'
    meg_data.save(preproc_save_path + preproc_meg_data_fname, overwrite=True)


