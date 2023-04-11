import os
import mne
import numpy as np
from paths import paths
import load
import setup
import pandas as pd
import functions_analysis


# --------- Setup ---------#
# Select subject to run
subject_code = 2
use_ica_data = True

# Load experiment info
exp_info = setup.exp_info()

# Define Subjects_dir as Freesurfer output folder
mri_path = paths().mri_path()
subjects_dir = os.path.join(mri_path, 'FreeSurfer_out')
os.environ["SUBJECTS_DIR"] = subjects_dir

# Digitalization data path
dig_path = paths().opt_path()


# --------- Coregistration ---------#
if use_ica_data:
    # Load subject and meg clean data
    subject = load.ica_subject(exp_info=exp_info, subject_code=subject_code)
    meg_data = load.ica_data(subject=subject)
    data_type = 'ICA'
else:
    subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)
    meg_data = subject.load_preproc_meg()
    data_type = 'RAW'

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
dig_info = meg_data.info.copy()
dig_info.set_montage(montage=dig_montage)

# Save raw instance with info
info_raw = mne.io.RawArray(np.zeros((dig_info['nchan'], 1)), dig_info)
dig_info_path = dig_path_subject + '/info_raw.fif'
info_raw.save(dig_info_path, overwrite=True)

# Align and save fiducials and transformation files to FreeSurfer/subject/bem folder
mne.gui.coregistration(subject=subject.subject_id, subjects_dir=subjects_dir, inst=dig_info_path)

# Check mean distances
trans_path = os.path.join(subjects_dir, subject.subject_id, 'bem', '{}-trans.fif'.format(subject.subject_id))
trans = mne.read_trans(trans_path)
print('Distance from head origin to MEG origin: %0.1f mm'
      % (1000 * np.linalg.norm(meg_data.info['dev_head_t']['trans'][:3, 3])))
print('Distance from head origin to MRI origin: %0.1f mm'
      % (1000 * np.linalg.norm(trans['trans'][:3, 3])))


# --------- Bem model ---------#
# Source data and models path
sources_path = paths().sources_path()
sources_path_subject = sources_path + subject.subject_id
os.makedirs(sources_path_subject, exist_ok=True)

model = mne.make_bem_model(subject=subject.subject_id, ico=5, conductivity=[0.3], subjects_dir=subjects_dir)
bem = mne.make_bem_solution(model)

# Save
fname_bem = sources_path_subject + f'/{subject.subject_id}_bem-sol.fif'
mne.write_bem_solution(fname_bem, bem, overwrite=True)
# Load
# bem = mne.read_bem_solution(fname_bem)

# --------- Background noise covariance ---------#
cov = functions_analysis.noise_cov(exp_info=exp_info, subject=subject, bads=meg_data.info['bads'], use_ica_data=use_ica_data)

# --------- Source space, forward model and inverse operator ---------#
# Define if whether surface or volume source space

volume = True
if volume:
    save_name = 'volume'
else:
    save_name = 'surface'

if not volume:
    # Surface
    # Source model
    src = mne.setup_source_space(subject=subject.subject_id, spacing='oct6', subjects_dir=subjects_dir)
    fname_src = sources_path_subject + f'/{subject.subject_id}_surface-src.fif'
    mne.write_source_spaces(fname_src, src, overwrite=True)

    # Forward model
    fwd = mne.make_forward_solution(meg_data.info, trans=trans_path, src=src, bem=bem)
    fname_fwd = sources_path_subject + f'/{subject.subject_id}_surface-fwd.fif'
    mne.write_forward_solution(fname_fwd, fwd, overwrite=True)

    # Inverse operator
    inv = mne.minimum_norm.make_inverse_operator(meg_data.info, fwd, cov)
    fname_inv = sources_path_subject + f'/{subject.subject_id}_surface-inv_{data_type}.fif'
    mne.minimum_norm.write_inverse_operator(fname_inv, inv, overwrite=True)

else:
    # Volume
    # Source model
    surface = subjects_dir + f'/{subject.subject_id}/bem/inner_skull.surf'
    vol_src = mne.setup_volume_source_space(subject=subject.subject_id, subjects_dir=subjects_dir, surface=surface,
                                            sphere_units='m', add_interpolator=True)
    fname_src = sources_path_subject + f'/{subject.subject_id}_volume-src.fif'
    mne.write_source_spaces(fname_src, vol_src, overwrite=True)

    # Forward model
    fwd_vol = mne.make_forward_solution(meg_data.info, trans=trans_path, src=vol_src, bem=bem)
    fname_fwd = sources_path_subject + f'/{subject.subject_id}_volume-fwd.fif'
    mne.write_forward_solution(fname_fwd, fwd_vol, overwrite=True)

    # Inverse operator
    inv_vol = mne.minimum_norm.make_inverse_operator(meg_data.info, fwd_vol, cov)
    fname_inv = sources_path_subject + f'/{subject.subject_id}_volume-inv_{data_type}.fif'
    mne.minimum_norm.write_inverse_operator(fname_inv, inv_vol, overwrite=True)