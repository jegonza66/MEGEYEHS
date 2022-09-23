import mne
import os
import numpy as np
import pickle

from paths import paths
import load
import preproc_plot
import preproc_functions


def preprocess(subject):

    #---------------- Load data ----------------#
    # Define subject
    subject = load.subject(subject)
    # Load Meg data
    raw = subject.ctf_data()

    # Get et channels by name [Gaze x, Gaze y, Pupils]
    et_channel_names = ['UADC001-4123', 'UADC002-4123', 'UADC013-4123']

    print('\nGetting ET channels data from MEG')
    et_channels_meg = raw.get_data(picks=et_channel_names)

    # Get separate data from et channels
    meg_gazex_data_raw = et_channels_meg[0]
    meg_gazey_data_raw = et_channels_meg[1]
    meg_pupils_data_raw = et_channels_meg[2]

    #---------------- Reescaling based on conversion parameters ----------------#
    meg_gazex_data_scaled, meg_gazey_data_scaled = preproc_functions.reescale_et_channels(meg_gazex_data_raw=meg_gazex_data_raw,
                                                                                          meg_gazey_data_raw=meg_gazey_data_raw)

    #---------------- Blinks removal ----------------#
    # Define intervals around blinks to also fill with nan. Due to conversion noise from square signal
    blink_min_dur = 70
    start_interval_samples = 12
    end_interval_samples = 24

    meg_gazex_data_clean, meg_gazey_data_clean, meg_pupils_data_clean = preproc_functions.blinks_to_nan(meg_gazex_data_scaled=meg_gazex_data_scaled,
                                                                                                        meg_gazey_data_scaled=meg_gazey_data_scaled,
                                                                                                        meg_pupils_data_raw=meg_pupils_data_raw,
                                                                                                        start_interval_samples=start_interval_samples,
                                                                                                        end_interval_samples=end_interval_samples)

    #---------------- Missing signal interpolation ----------------#
    et_channels_meg = preproc_functions.fake_blink_interpolate(meg_gazex_data_clean=meg_gazex_data_clean,
                                                               meg_gazey_data_clean=meg_gazey_data_clean,
                                                               meg_pupils_data_clean=meg_pupils_data_clean,
                                                               blink_min_dur=blink_min_dur,
                                                               start_interval_samples=start_interval_samples,
                                                               end_interval_samples=start_interval_samples,
                                                               sfreq=raw.info['sfreq'])

    #---------------- Defining response events and trials ----------------#
    bh_data, raw, subject = preproc_functions.define_events_trials(raw=raw, subject=subject)

    #---------------- Fixations and saccades detection ----------------#
    fixations, saccades = preproc_functions.fixations_saccades_detection(raw=raw, meg_gazex_data_clean=meg_gazex_data_clean,
                                                                         meg_gazey_data_clean=meg_gazey_data_clean,
                                                                         subject=subject, force_run=True)

    #---------------- Fixations classification ----------------#
    fixations = preproc_functions.fixation_classification(bh_data=bh_data, fixations=fixations, raw=raw,
                                                          meg_pupils_data_clean=meg_pupils_data_clean)

    #---------------- Items classification ----------------#
    fixations_vs, items_pos = preproc_functions.target_vs_distractor(fixations=fixations, bh_data=bh_data,
                                                                     distance_threshold=100)

    #---------------- Save fix time distribution, pupils size vs mss, scanpath and trial gaze figures ----------------#
    preproc_plot.first_fixation_delay(fixations=fixations, subject=subject)
    preproc_plot.pupil_size_increase(fixations=fixations, response_trials_meg=raw.annotations.trial, subject=subject)
    preproc_plot.performance(subject=subject)

    print('Plotting scanpaths and trials gaze screens')
    for trial in raw.annotations.trial:
        print(f'\rTrial {trial}', end='')

        preproc_plot.scanpath(fixations_vs=fixations_vs, items_pos=items_pos, bh_data=bh_data, raw=raw,
                              gazex=meg_gazex_data_clean, gazey=meg_gazey_data_clean, subject=subject, trial=trial)

        preproc_plot.trial_gaze(raw=raw, bh_data=bh_data, gazex=meg_gazex_data_clean, gazey=meg_gazey_data_clean,
                                subject=subject, trial=trial)

    #---------------- Add scaled data to meg data ----------------#
    print('\nSaving scaled et data to meg raw data structure')
    # copy raw structure
    raw_et = raw.copy()
    # make new raw structure from et channels only
    raw_et = mne.io.RawArray(et_channels_meg, raw_et.pick(et_channel_names).info)
    # change channel names
    for ch_name, new_name in zip(raw_et.ch_names, ['ET_gaze_x', 'ET_gaze_y', 'ET_pupils']):
        raw_et.rename_channels({ch_name: new_name})

    # Pick data from MEG channels and other channels of interest
    channel_indices = mne.pick_types(raw.info, meg=True)
    channel_indices = np.append(channel_indices, mne.pick_channels(raw.info['ch_names'], ['UPPT001']))
    raw.pick(channel_indices)

    # save to original raw structure (requires to load data)
    print('Loading MEG data')
    raw.load_data()
    print('Adding new ET channels')
    raw.add_channels([raw_et])
    del (raw_et)
    # Correct raw.annotations length
    raw.annotations.ch_names = raw.annotations.ch_names[:len(raw.annotations.description)]
    raw.annotations.duration = raw.annotations.duration[:len(raw.annotations.description)]

    #---------------- Save preprocesed data ----------------#
    print('Saving preprocessed data')
    # Path
    preproc_data_path = paths().preproc_path()
    preproc_save_path = preproc_data_path + subject.subject_id + '/'
    os.makedirs(preproc_save_path, exist_ok=True)

    # Add data to subject class
    subject.bh_data = bh_data
    subject.fixations = fixations
    subject.fixations_vs = fixations_vs
    subject.saccades = saccades

    f = open(preproc_save_path + 'Subject_data.pkl', 'wb')
    pickle.dump(subject, f)
    f.close()

    # # Save fixations
    # fixations.to_csv(preproc_save_path + 'fixations.csv')
    # fixations_vs.to_csv(preproc_save_path + 'fixations_vs.csv')

    # Save MEG
    preproc_meg_data_fname = f'Subject_{subject.subject_id}_meg.fif'
    raw.save(preproc_save_path + preproc_meg_data_fname, overwrite=True)

    # Save events
    # evt = mne.events_from_annotations(raw)

    print(f'Preprocessed data saved to {preproc_save_path}')


for subject in [2, 3, 4, 5]:
    preprocess(subject)
