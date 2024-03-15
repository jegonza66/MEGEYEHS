import os
import numpy as np
from osl_dynamics.data import Data
import osl_dynamics.models
from osl_dynamics.models.dynemo import Config, Model
import setup
from paths import paths
import mne
import functions_analysis
import load

# Define Subjects_dir as Freesurfer output folder
subjects_dir = os.path.join(paths().mri_path(), 'FreeSurfer_out')
os.environ["SUBJECTS_DIR"] = subjects_dir

exp_info = setup.exp_info()

all_trials = False
epoch_id = 'ms'
mss = None
use_ica_data = True
dsfreq = 250
l_freq, h_freq = None, 100
filter_method = 'iir'

# Source estimation parameters
force_fsaverage = False
model_name = 'lcmv'
surf_vol = 'surface'
ico = 4
spacing = 10.
pick_ori = None  # 'vector' For dipoles, 'max-power' for fixed dipoles in the direction tha maximizes output power
# Parcelation (aparc / aparc.a2009s)
parcelation = 'aparc'

training_data_paths = []
for subject_code in exp_info.subjects_ids:

    # Load subject
    if use_ica_data:
        subject = load.ica_subject(exp_info=exp_info, subject_code=subject_code)
    else:
        subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)

    # Source data path
    sources_path_subject = paths().sources_path() + subject.subject_id
    sources_labels_ts_fame = f'_{model_name}_{surf_vol}_ico{ico}_{int(spacing)}_{pick_ori}_{parcelation}'

    try:
        # Load data in OSL
        training_data = Data(sources_path_subject + sources_labels_ts_fame)

    except:
        # Downsmpled sensor data
        downsampled_path = paths().save_path() + f'Downsampled_Data/{epoch_id}/{dsfreq}_l{l_freq}_h{h_freq}_{filter_method}/'
        os.makedirs(downsampled_path, exist_ok=True)
        downsampled_fname = downsampled_path + f'{subject.subject_id}_raw.fif'

        try:
            # Load downsampled data
            cont_data = mne.io.read_raw_fif(downsampled_fname, preload=True)

        except:
            # Downsample and filter data
            if use_ica_data:
                meg_data = load.ica_data(subject=subject)
            else:
                meg_data = subject.load_preproc_meg_data()

            # Downsample data
            meg_data.resample(sfreq=dsfreq)

            # Filter data
            meg_data = meg_data.filter(l_freq=l_freq, h_freq=h_freq, method=filter_method)

            if all_trials:
                # Extract all trial data
                all_events, all_event_id = mne.events_from_annotations(meg_data, verbose=False)
                all_event_numbers = {v: k for k, v in all_event_id.items()}
                trial_durations = [vs_end - cross_1 for (vs_end, cross_1) in zip(subject.vsend, subject.cross1)]
                ms_durations = subject.bh_data.stDur.values() + 0.75

                epochs = []
                for trial_idx in range(len(ms_durations)):
                    trial = str(trial_idx + 1)
                    event_id = f'ms'
                    epoch, event = functions_analysis.epoch_data(subject=subject, mss=mss, corr_ans=None, tgt_pres=None,
                                                                   epoch_id=event_id, meg_data=meg_data, tmin=0, tmax=ms_durations[trial_idx], trial_num=[trial], trial_dur=None,
                                                                   reject=False, baseline=(None, None), save_data=False, epochs_save_path=None,
                                                                   epochs_data_fname=None)
                    epochs.append(epoch)

                # Get data as array to use in dynemo
                epochs_data = [epoch.get_data() for epoch in epochs]
                epochs_array = np.zeros((epoch.get_data().shape[1], sum(epoch.get_data().shape[2] for epoch in epochs)))

                # Make concatenated trial data
                for idx, epoch in enumerate(epochs):
                    if idx == 0:
                        idx = 1
                    epochs_array[:, len(epochs[idx - 1].times)]
                cont_data = mne.io.RawArray(epochs_array, info=meg_data.info)

                # Save
                cont_data.save(downsampled_fname, overwrite=True)

            else:
                # Epoch data
                epochs, events = functions_analysis.epoch_data(subject=subject, mss=mss, corr_ans=None, trial_dur=None,
                                                               tgt_pres=None, baseline=(None, None), reject=False,
                                                               epoch_id=epoch_id, meg_data=meg_data, tmin=-0.75, tmax=2,
                                                               save_data=False, epochs_save_path=None,
                                                               epochs_data_fname=None)

                # Get data as array to use in dynemo
                epochs_data = epochs.get_data()
                epochs_data = epochs_data.swapaxes(0,1)
                epochs_array = epochs_data.reshape((epochs_data.shape[0], epochs_data.shape[2]*epochs_data.shape[1]))

                cont_data = mne.io.RawArray(epochs_array, info=meg_data.info)

                # Save
                cont_data.save(downsampled_fname, overwrite=True)


        # Load forward model
        if surf_vol == 'volume':
            fname_fwd = sources_path_subject + f'/{subject_code}_volume_ico{ico}_{int(spacing)}-fwd.fif'
        elif surf_vol == 'surface':
            fname_fwd = sources_path_subject + f'/{subject_code}_surface_ico{ico}-fwd.fif'
        elif surf_vol == 'mixed':
            fname_fwd = sources_path_subject + f'/{subject_code}_mixed_ico{ico}_{int(spacing)}-fwd.fif'
        fwd = mne.read_forward_solution(fname_fwd)
        src = fwd['src']

        # Load filter
        if surf_vol == 'volume':
            fname_filter = sources_path_subject + f'/{subject_code}_volume_ico{ico}_{int(spacing)}_{pick_ori}-{model_name}.fif'
        elif surf_vol == 'surface':
            fname_filter = sources_path_subject + f'/{subject_code}_surface_ico{ico}_{pick_ori}-{model_name}.fif'
        elif surf_vol == 'mixed':
            fname_filter = sources_path_subject + f'/{subject_code}_mixed_ico{ico}_{int(spacing)}_{pick_ori}-{model_name}.fif'
        filters = mne.beamformer.read_beamformer(fname_filter)

        # Compute source estimation
        stc = mne.beamformer.apply_lcmv_raw(raw=cont_data, filters=filters)

        # Get labels for FreeSurfer 'aparc' cortical parcellation with 34 labels/hemi
        fsaverage_labels = mne.read_labels_from_annot(subject='fsaverage', parc=parcelation, subjects_dir=subjects_dir)
        # Remove 'unknown' label for fsaverage aparc labels
        if parcelation == 'aparc':
            print("Dropping extra 'unkown' label from lh.")
            drop_idxs = [i for i, label in enumerate(fsaverage_labels) if 'unknown' in label.name]
            for drop_idx in drop_idxs:
                fsaverage_labels.pop(drop_idx)

        if surf_vol == 'volume':
            labels = subjects_dir + f'/{subject_code}/mri/aparc+aseg.mgz'
        elif subject_code != 'fsaverage':
            # Get labels for FreeSurfer cortical parcellation
            labels = mne.read_labels_from_annot(subject=subject_code, parc=parcelation, subjects_dir=subjects_dir)
        else:
            labels = fsaverage_labels

        # Average the source estimates within each label using sign-flips to reduce signal cancellations
        label_ts = mne.extract_label_time_course(stcs=stc, labels=labels, src=src, mode='auto', return_generator=False)

        training_data = Data(label_ts, sampling_frequency=dsfreq)

        methods = {
            "filter": {"low_freq": 7, "high_freq": 13},  # study the alpha-band
            "amplitude_envelope": {},
            "moving_average": {"n_window": 5},
            "standardize": {},
        }
        training_data.prepare(methods)

    config = Config(
        n_modes=6,
        n_channels=80,
        sequence_length=200,
        inference_n_units=64,
        inference_normalization="layer",
        model_n_units=64,
        model_normalization="layer",
        learn_alpha_temperature=True,
        initial_alpha_temperature=1.0,
        learn_means=False,
        learn_covariances=True,
        do_kl_annealing=True,
        kl_annealing_curve="tanh",
        kl_annealing_sharpness=10,
        n_kl_annealing_epochs=50,
        batch_size=16,
        learning_rate=0.01,
        n_epochs=10,
    )

    model = Model(config)
    model.compile(run_eagerly=True)

    init_history = model.random_subset_initialization(training_data, n_epochs=1, n_init=3, take=0.2)

    history = model.fit(training_data)