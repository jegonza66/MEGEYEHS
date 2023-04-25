import pathlib
import os
import mne
import save
import setup
from paths import paths
import load
exp_info = setup.exp_info()

subjects_dir = os.path.join(paths().mri_path(), 'FreeSurfer_out')

for subject_code in exp_info.subjects_ids:
# for subject_code in ['16191001']:

    # Load subject
    subject_preproc = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)
    subject_ica = load.ica_subject(exp_info=exp_info, subject_code=subject_code)

    print('\nLoading MEG data')
    # get subject path
    ctf_path = pathlib.Path(os.path.join(paths().ctf_path(), subject_preproc.subject_id))
    ds_files = list(ctf_path.glob('*{}*.ds'.format(subject_preproc.subject_id)))
    ds_files.sort()

    # Use fsaverage MRI
    trans_path = os.path.join(subjects_dir, 'fsaverage', 'bem', 'fsaverage-trans.fif')
    fids_path = os.path.join(subjects_dir, 'fsaverage', 'bem', 'fsaverage-fiducials.fif')

    # Load sesions
    # If more than 1 session concatenate all data to one raw data
    if len(ds_files) > 1:
        raws_list = []
        for i in range(len(ds_files)):
            raw = mne.io.read_raw_ctf(ds_files[i], system_clock='ignore')
            raws_list.append(raw)

            surfaces = dict(brain=0.7, outer_skull=0.5, head=0.4)
            # Try plotting with head skin and brain
            try:
                mne.viz.plot_alignment(raw.info, trans=trans_path, subject='fsaverage',
                                       subjects_dir=subjects_dir, surfaces=surfaces,
                                       show_axes=True, dig=False, eeg=[], meg='sensors',
                                       coord_frame='meg', mri_fiducials=fids_path)
            # Plot only outer skin
            except:
                mne.viz.plot_alignment(raw.info, trans=trans_path, subject='fsaverage',
                                       subjects_dir=subjects_dir, surfaces='outer_skin',
                                       show_axes=True, dig=False, eeg=[], meg='sensors',
                                       coord_frame='meg', mri_fiducials=fids_path)

        # MEG data structure
        raw = mne.io.concatenate_raws(raws_list, on_mismatch='warn')

        head_loc_idx = int(input('Already chosen a head localization transform? Enter the index number to continue'))

        # Add head loc idx to subject
        subject_preproc.head_loc_idx = head_loc_idx
        subject_ica.head_loc_idx = head_loc_idx

        # Save subjects
        preproc_path = paths().preproc_path() + subject_preproc.subject_id + '/'
        save.var(var=subject_preproc, path=preproc_path, fname='Subject_data.pkl')

        ica_path = paths().ica_path() + subject_ica.subject_id + '/'
        save.var(var=subject_ica, path=ica_path, fname='Subject_data.pkl')

        # Change dev_head_t in preproc data
        meg_data_preproc = subject_preproc.load_preproc_meg_data()
        meg_data_preproc.info['dev_head_t'] = raws_list[head_loc_idx].info['dev_head_t']

        # Change dev_head_t in ica data
        meg_data_ica = subject_ica.load_ica_meg_data()
        meg_data_ica.info['dev_head_t'] = raws_list[head_loc_idx].info['dev_head_t']

        # Save preproc MEG
        preproc_meg_data_fname = f'Subject_{subject_preproc.subject_id}_meg.fif'
        raw.save(preproc_path + preproc_meg_data_fname, overwrite=True)

        # Save ICA MEG
        ica_meg_data_fname = f'Subject_{subject_ica.subject_id}_ICA.fif'
        raw.save(ica_path + ica_meg_data_fname, overwrite=True)


    # If only one session return that session as whole raw data
    elif len(ds_files) == 1:
        raw = mne.io.read_raw_ctf(ds_files[0], system_clock='ignore')

        # Add head loc idx to subject
        subject_preproc.head_loc_idx = 0
        subject_ica.head_loc_idx = 0

        # Save
        preproc_path = paths().preproc_path() + subject_preproc.subject_id + '/'
        save.var(var=subject_preproc, path=preproc_path, fname='Subject_data.pkl')

        ica_path = paths().ica_path() + subject_ica.subject_id + '/'
        save.var(var=subject_ica, path=ica_path, fname='Subject_data.pkl')
