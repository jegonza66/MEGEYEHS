import os
import mne
import numpy as np
from paths import paths
import load
import setup
import pandas as pd
import functions_analysis
import mne.beamformer as beamformer

# Load experiment info
exp_info = setup.exp_info()


# --------- Setup ---------#

# Define surface or volume source space
surf_vol = 'volume'
use_ica_data = True
force_fsaverage = False
ico = 5
spacing = 10.
pick_ori = None
high_freq = True

# Define Subjects_dir as Freesurfer output folder
mri_path = paths().mri_path()
subjects_dir = os.path.join(mri_path, 'FreeSurfer_out')
os.environ["SUBJECTS_DIR"] = subjects_dir

# Digitalization data path
dig_path = paths().opt_path()


# --------- Coregistration ---------#

# Iterate over subjects
for subject_code in exp_info.subjects_ids:

    if use_ica_data:
        # Load subject and meg clean data
        subject = load.ica_subject(exp_info=exp_info, subject_code=subject_code)
        if high_freq:
            meg_data = load.filtered_data(subject=subject, band_id='HGamma', save_data=False)
        else:
            meg_data = load.ica_data(subject=subject)
        data_type = 'ICA'
    else:
        subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)
        if high_freq:
            meg_data = load.filtered_data(subject=subject, band_id='HGamma', use_ica_data=False,  save_data=False)
        else:
            meg_data = subject.load_preproc_meg_data()
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
    sources_path_fsaverage = sources_path + 'fsaverage'
    os.makedirs(sources_path_subject, exist_ok=True)
    os.makedirs(sources_path_fsaverage, exist_ok=True)

    fname_bem = sources_path + subject_code + f'/{subject_code}_bem_ico{ico}-sol.fif'
    try:
        # Load
        bem = mne.read_bem_solution(fname_bem)
    except:
        # Compute
        model = mne.make_bem_model(subject=subject_code, ico=ico, conductivity=[0.3], subjects_dir=subjects_dir)
        bem = mne.make_bem_solution(model)
        # Save
        mne.write_bem_solution(fname_bem, bem, overwrite=True)

    # --------- Background noise covariance ---------#
    noise_cov = functions_analysis.noise_cov(exp_info=exp_info, subject=subject, bads=meg_data.info['bads'], use_ica_data=use_ica_data, high_freq=high_freq)

    # # Extra
    # # Change head loc
    # head_loc_idx = 1
    # meg_data.info['dev_head_t'] = raws_list[head_loc_idx].info['dev_head_t']

    # --------- Raw data covariance ---------#
    # Pick meg channels for source modeling
    meg_data.pick('meg')

    # Compute covariance to withdraw from meg data
    data_cov = mne.compute_raw_covariance(meg_data, reject=dict(mag=4e-12), rank=None)

    # --------- Source space, forward model and inverse operator ---------#
    if surf_vol == 'volume':
        # Volume
        # Source model
        fname_src = sources_path + subject_code + f'/{subject_code}_volume_ico{ico}_{int(spacing)}-src.fif'
        try:
            # Load
            src = mne.read_source_spaces(fname_src)
        except:
            # Compute
            src = mne.setup_volume_source_space(subject=subject_code, subjects_dir=subjects_dir, bem=bem, pos=spacing,
                                                sphere_units='m', add_interpolator=True)
            # Save
            mne.write_source_spaces(fname_src, src, overwrite=True)

        # Forward model
        fname_fwd = sources_path_subject + f'/{subject_code}_volume_ico{ico}_{int(spacing)}-fwd.fif'
        try:
            # Load
            fwd = mne.read_forward_solution(fname=fname_fwd)
        except:
            # Compute
            fwd = mne.make_forward_solution(meg_data.info, trans=trans_path, src=src, bem=bem)
            mne.write_forward_solution(fname_fwd, fwd, overwrite=True)

        # Spatial filter
        rank = sum([ch_type == 'mag' for ch_type in meg_data.get_channel_types()]) - len(meg_data.info['bads'])
        if use_ica_data:
            rank -= len(subject.ex_components)

        # Define linearly constrained minimum variance spatial filter
        # reg parameter is for regularization on rank deficient matrices (rank < channels)
        filters = beamformer.make_lcmv(info=meg_data.info, forward=fwd, data_cov=data_cov, reg=0.05,
                                       noise_cov=noise_cov, pick_ori=pick_ori, rank=dict(mag=rank))

        # Save
        if high_freq:
            fname_lmcv = sources_path_subject + f'/{subject_code}_volume_ico{ico}_{int(spacing)}_{pick_ori}_hfreq-lcmv.fif'
        else:
            fname_lmcv = sources_path_subject + f'/{subject_code}_volume_ico{ico}_{int(spacing)}_{pick_ori}-lcmv.fif'
        filters.save(fname=fname_lmcv, overwrite=True)

    elif surf_vol == 'surface':
        # Surface
        # Source model
        fname_src = sources_path + subject_code + f'/{subject_code}_surface_ico{ico}-src.fif'
        try:
            # Load
            src = mne.read_source_spaces(fname_src)
        except:
            # Compute
            src = mne.setup_source_space(subject=subject_code, spacing=f'ico{ico}', subjects_dir=subjects_dir)
            # Save
            mne.write_source_spaces(fname_src, src, overwrite=True)

        # Forward model
        fwd = mne.make_forward_solution(meg_data.info, trans=trans_path, src=src, bem=bem)
        fname_fwd = sources_path_subject + f'/{subject_code}_surface_ico{ico}-fwd.fif'
        mne.write_forward_solution(fname_fwd, fwd, overwrite=True)

        # Spatial filter
        rank = sum([ch_type == 'mag' for ch_type in meg_data.get_channel_types()]) - len(meg_data.info['bads'])
        if use_ica_data:
            rank -= len(subject.ex_components)

        # Define linearly constrained minimum variance spatial filter
        # reg parameter is for regularization on rank deficient matrices (rank < channels)
        filters = beamformer.make_lcmv(info=meg_data.info, forward=fwd, data_cov=data_cov, reg=0.05,
                                       noise_cov=noise_cov, pick_ori=pick_ori, rank=dict(mag=rank))

        # Save
        if high_freq:
            fname_lmcv = sources_path_subject + f'/{subject_code}_surface_ico{ico}_{pick_ori}_hfreq-lcmv.fif'
        else:
            fname_lmcv = sources_path_subject + f'/{subject_code}_surface_ico{ico}_{pick_ori}-lcmv.fif'
        filters.save(fname=fname_lmcv, overwrite=True)

    elif surf_vol == 'mixed':
        fname_src_mix = sources_path + subject_code + f'/{subject_code}_mixed_ico{ico}_{int(spacing)}-src.fif'
        try:
            # Load
            src_surf = mne.read_source_spaces(fname_src_surf)
        except:
            # Mixed
            # Surface source model
            fname_src_surf = sources_path + subject_code + f'/{subject_code}_surface_ico{ico}-src.fif'
            try:
                # Load
                src_surf = mne.read_source_spaces(fname_src_surf)
            except:
                # Compute
                src_surf = mne.setup_source_space(subject=subject_code, spacing=f'ico{ico}', subjects_dir=subjects_dir)
                # Save
                mne.write_source_spaces(fname_src_surf, src_surf, overwrite=True)

            # Volume source model
            fname_src_vol = sources_path + subject_code + f'/{subject_code}_volume_ico{ico}_{int(spacing)}-src.fif'
            try:
                # Load
                src_vol = mne.read_source_spaces(fname_src_vol)
            except:
                # Compute
                src_vol = mne.setup_volume_source_space(subject=subject_code, subjects_dir=subjects_dir, bem=bem, pos=spacing, sphere_units='m', add_interpolator=True)
                # Save
                mne.write_source_spaces(fname_src_vol, src_vol, overwrite=True)

            # Mixed source space
            src = src_surf + src_vol
            # Save
            mne.write_source_spaces(fname_src_mix, src, overwrite=True)

        # Forward model
        fwd = mne.make_forward_solution(meg_data.info, trans=trans_path, src=src, bem=bem)
        fname_fwd = sources_path_subject + f'/{subject_code}_mixed_ico{ico}_{int(spacing)}-fwd.fif'
        mne.write_forward_solution(fname_fwd, fwd, overwrite=True)

        # Spatial filter
        rank = sum([ch_type == 'mag' for ch_type in meg_data.get_channel_types()]) - len(meg_data.info['bads'])
        if use_ica_data:
            rank -= len(subject.ex_components)

        # Define linearly constrained minimum variance spatial filter
        # reg parameter is for regularization on rank deficient matrices (rank < channels)
        filters = beamformer.make_lcmv(info=meg_data.info, forward=fwd, data_cov=data_cov, reg=0.05,
                                       noise_cov=noise_cov, pick_ori=pick_ori, rank=dict(mag=rank))

        # Save
        if high_freq:
            fname_lmcv = sources_path_subject + f'/{subject_code}_mixed_ico{ico}_{int(spacing)}_{pick_ori}_hfreq-lcmv.fif'
        else:
            fname_lmcv = sources_path_subject + f'/{subject_code}_mixed_ico{ico}_{int(spacing)}_{pick_ori}-lcmv.fif'
        filters.save(fname=fname_lmcv, overwrite=True)
