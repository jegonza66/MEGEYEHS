import pathlib
import os
import mne
import setup
from paths import paths
import plot_general
import load
exp_info = setup.exp_info()

subjects_dir = os.path.join(paths().mri_path(), 'FreeSurfer_out')

# for subject_code in exp_info.subjects_ids:
for subject_code in ['16191001']:

    subject = load.ica_subject(exp_info=exp_info, subject_code=subject_code)

    print('\nLoading MEG data')
    # get subject path
    ctf_path = pathlib.Path(os.path.join(paths().ctf_path(), subject.subject_id))
    ds_files = list(ctf_path.glob('*{}*.ds'.format(subject.subject_id)))
    ds_files.sort()

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

    # If only one session return that session as whole raw data
    elif len(ds_files) == 1:
        raw = mne.io.read_raw_ctf(ds_files[0], system_clock='ignore')

    # Missing data
    else:
        raise ValueError('No .ds files found in subject directory: {}'.format(subject.subj_path))