import pathlib
import os
import mne
import save
import setup
from paths import paths
import load
exp_info = setup.exp_info()

subjects_dir = os.path.join(paths().mri_path(), 'FreeSurfer_out')
subjects_ids = ['15909001', '15912001', '15910001', '15950001', '15911001', '11535009', '16191001', '16200001',
                '16201001', '09991040', '10925091', '16263002', '16269001']

for subject_code in subjects_ids:
# for subject_code in ['16191001']:

    # Load subject
    subject_preproc = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)
    print(subject_preproc.head_loc_idx)

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



        head_loc_idx = int(input('Already chosen a head localization transform? Enter the index number to continue'))

        # Add head loc idx to subject
        subject_preproc.head_loc_idx = head_loc_idx
        subject_ica.head_loc_idx = head_loc_idx

        # Save subjects
        preproc_path = paths().preproc_path() + subject_preproc.subject_id + '/'
        save.var(var=subject_preproc, path=preproc_path, fname='Subject_data.pkl')

        ica_path = paths().ica_path() + subject_ica.subject_id + '/'
        save.var(var=subject_ica, path=ica_path, fname='Subject_data.pkl')

        if head_loc_idx:

            # Change dev_head_t in preproc data
            meg_data_preproc = subject_preproc.load_preproc_meg_data(preload=True)
            meg_data_preproc.info['dev_head_t'] = raws_list[head_loc_idx].info['dev_head_t']
            # Save preproc MEG
            preproc_meg_data_fname = f'Subject_{subject_preproc.subject_id}_meg.fif'
            meg_data_preproc.save(preproc_path + preproc_meg_data_fname, overwrite=True)
            del meg_data_preproc

            # Change dev_head_t in ica data
            meg_data_ica = subject_ica.load_ica_meg_data(preload=True)
            meg_data_ica.info['dev_head_t'] = raws_list[head_loc_idx].info['dev_head_t']
            # Save ICA MEG
            ica_meg_data_fname = f'Subject_{subject_ica.subject_id}_ICA.fif'
            meg_data_ica.save(ica_path + ica_meg_data_fname, overwrite=True)
            del meg_data_ica

    # If only one session return that session as whole raw data
    elif len(ds_files) == 1:

        # Add head loc idx to subject
        subject_preproc.head_loc_idx = 0
        subject_ica.head_loc_idx = 0

        # Save
        preproc_path = paths().preproc_path() + subject_preproc.subject_id + '/'
        save.var(var=subject_preproc, path=preproc_path, fname='Subject_data.pkl')

        ica_path = paths().ica_path() + subject_ica.subject_id + '/'
        save.var(var=subject_ica, path=ica_path, fname='Subject_data.pkl')
