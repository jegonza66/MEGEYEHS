import os
import mne
import numpy as np
from paths import paths
import load
import setup
import pandas as pd
import functions_analysis

# Load experiment info
exp_info = setup.exp_info()

# --------- Setup ---------#
subjects = ['15909001', '15910001', '15950001', '15911001', '16191001', '16263002']

subjects_ids = ['15909001', '15912001', '15910001', '15950001', '15911001', '11535009', '16191001', '16200001',
                '16201001', '10925091', '16263002', '16269001']

subjects = ['16263002']

# Define surface or volume source space
volume = True
use_ica_data = True
force_fsaverage = False

# Define Subjects_dir as Freesurfer output folder
mri_path = paths().mri_path()
subjects_dir = os.path.join(mri_path, 'FreeSurfer_out')
os.environ["SUBJECTS_DIR"] = subjects_dir

# Digitalization data path
dig_path = paths().opt_path()

# --------- Coregistration ---------#
# Iterate over subjects
for subject_code in subjects:

    if use_ica_data:
        # Load subject and meg clean data
        subject = load.ica_subject(exp_info=exp_info, subject_code=subject_code)
        meg_data = subject.load_ica_meg_data()
        data_type = 'ICA'
    else:
        subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)
        meg_data_orig = subject.load_preproc_meg_data()
        data_type = 'RAW'

    if force_fsaverage:
        subject_code = 'fsaverage'
        # Check mean distances if already run transformation
        trans_path = os.path.join(subjects_dir, subject_code, 'bem', f'{subject_code}-trans.fif')
        trans = mne.read_trans(trans_path)
        print('Distance from head origin to MEG origin: %0.1f mm'
              % (1000 * np.linalg.norm(meg_data.info['dev_head_t']['trans'][:3, 3])))
        print('Distance from head origin to MRI origin: %0.1f mm'
              % (1000 * np.linalg.norm(trans['trans'][:3, 3])))

    else:
        # Check if subject has MRI data
        try:
            fs_subj_path = os.path.join(subjects_dir, subject.subject_id)
            os.listdir(fs_subj_path)
            try:
                # Check mean distances if already run transformation
                trans_path = os.path.join(subjects_dir, subject_code, 'bem', f'{subject_code}-trans.fif')
                trans = mne.read_trans(trans_path)
                print('Distance from head origin to MEG origin: %0.1f mm'
                      % (1000 * np.linalg.norm(meg_data.info['dev_head_t']['trans'][:3, 3])))
                print('Distance from head origin to MRI origin: %0.1f mm'
                      % (1000 * np.linalg.norm(trans['trans'][:3, 3])))

            except:
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
                dig_info = meg_data.pick('meg').info.copy()
                dig_info.set_montage(montage=dig_montage)

                # Save raw instance with info
                info_raw = mne.io.RawArray(np.zeros((dig_info['nchan'], 1)), dig_info)
                dig_info_path = dig_path_subject + '/info_raw.fif'
                info_raw.save(dig_info_path, overwrite=True)

                # Align and save fiducials and transformation files to FreeSurfer/subject/bem folder
                mne.gui.coregistration(subject=subject.subject_id, subjects_dir=subjects_dir, inst=dig_info_path, block=True)

        # If subject has no MRI data
        except:
            subject_code = 'fsaverage'
            # Check mean distances if already run transformation
            trans_path = os.path.join(subjects_dir, subject_code, 'bem', f'{subject_code}-trans.fif')
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

    fname_bem = sources_path_subject + f'/{subject_code}_bem-sol.fif'
    try:
        # Load
        bem = mne.read_bem_solution(fname_bem)

    except:
        # Compute
        model = mne.make_bem_model(subject=subject_code, ico=5, conductivity=[0.3], subjects_dir=subjects_dir)
        bem = mne.make_bem_solution(model)

        # Save
        mne.write_bem_solution(fname_bem, bem, overwrite=True)

    # --------- Background noise covariance ---------#
    cov = functions_analysis.noise_cov(exp_info=exp_info, subject=subject, bads=meg_data.info['bads'], use_ica_data=use_ica_data)

    # --------- Source space, forward model and inverse operator ---------#
    if volume:
        # Volume
        # Source model
        surface = subjects_dir + f'/{subject_code}/bem/inner_skull.surf'
        vol_src = mne.setup_volume_source_space(subject=subject_code, subjects_dir=subjects_dir, surface=surface,
                                                sphere_units='m', add_interpolator=True)
        fname_src = sources_path_subject + f'/{subject_code}_volume-src.fif'
        mne.write_source_spaces(fname_src, vol_src, overwrite=True)

        # Forward model
        fwd_vol = mne.make_forward_solution(meg_data.info, trans=trans_path, src=vol_src, bem=bem)
        fname_fwd = sources_path_subject + f'/{subject_code}_volume-fwd.fif'
        mne.write_forward_solution(fname_fwd, fwd_vol, overwrite=True)

        # Inverse operator
        inv_vol = mne.minimum_norm.make_inverse_operator(meg_data.info, fwd_vol, cov)
        fname_inv = sources_path_subject + f'/{subject_code}_volume-inv_{data_type}.fif'
        mne.minimum_norm.write_inverse_operator(fname_inv, inv_vol, overwrite=True)

    else:
        #Surface
        # Source model
        src = mne.setup_source_space(subject=subject.subject_id, spacing='oct6', subjects_dir=subjects_dir)
        fname_src = sources_path_subject + f'/{subject_code}_surface-src.fif'
        mne.write_source_spaces(fname_src, src, overwrite=True)

        # Forward model
        fwd = mne.make_forward_solution(meg_data.info, trans=trans_path, src=src, bem=bem)
        fname_fwd = sources_path_subject + f'/{subject_code}_surface-fwd.fif'
        mne.write_forward_solution(fname_fwd, fwd, overwrite=True)

        # Inverse operator
        inv = mne.minimum_norm.make_inverse_operator(meg_data.info, fwd, cov)
        fname_inv = sources_path_subject + f'/{subject_code}_surface-inv_{data_type}.fif'
        mne.minimum_norm.write_inverse_operator(fname_inv, inv, overwrite=True)
