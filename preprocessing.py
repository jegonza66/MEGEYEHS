import setup
import load
import save
import preproc_plot
import preproc_functions
from paths import paths


def preprocess(subject_code, plot=False):

    #---------------- Load data ----------------#
    # Load experiment info
    exp_info = setup.exp_info()

    # Load run configuration
    config = load.config(path=paths().config_path(), fname='config.pkl')

    # Define subject
    subject = setup.raw_subject(exp_info=exp_info, config=config, subject_code=subject_code)

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
    meg_gazex_data_clean, meg_gazey_data_clean, meg_pupils_data_clean = preproc_functions.blinks_to_nan(meg_gazex_data_scaled=meg_gazex_data_scaled,
                                                                                                        meg_gazey_data_scaled=meg_gazey_data_scaled,
                                                                                                        meg_pupils_data_raw=meg_pupils_data_raw,
                                                                                                        config=subject.config.preproc)

    #---------------- Missing signal interpolation ----------------#
    et_channels_meg = preproc_functions.fake_blink_interpolate(meg_gazex_data_clean=meg_gazex_data_clean,
                                                               meg_gazey_data_clean=meg_gazey_data_clean,
                                                               meg_pupils_data_clean=meg_pupils_data_clean,
                                                               config=config.preprocessing, sfreq=raw.info['sfreq'])

    #---------------- Defining response events and trials ----------------#
    bh_data, raw, subject = preproc_functions.define_events_trials(raw=raw, subject=subject, config=config, exp_info=exp_info,
                                                                   et_channel_names=et_channel_names)

    #---------------- Fixations and saccades detection ----------------#
    fixations, saccades = preproc_functions.fixations_saccades_detection(raw=raw, meg_gazex_data_clean=meg_gazex_data_clean,
                                                                         meg_gazey_data_clean=meg_gazey_data_clean,
                                                                         subject=subject)

    # ---------------- Saccades classification ----------------#
    saccades, raw, subject = preproc_functions.saccades_classification(subject=subject, bh_data=bh_data, saccades=saccades, raw=raw)

    #---------------- Fixations classification ----------------#
    fixations, raw = preproc_functions.fixation_classification(subject=subject, bh_data=bh_data, fixations=fixations, saccades=saccades,
                                                               raw=raw, meg_pupils_data_clean=meg_pupils_data_clean)

    #---------------- Items classification ----------------#
    fixations, raw, items_pos = preproc_functions.target_vs_distractor(fixations=fixations, bh_data=bh_data,
                                                                       raw=raw, distance_threshold=100)

    #---------------- Save fix time distribution, pupils size vs mss, scanpath and trial gaze figures ----------------#
    if plot:
        preproc_plot.first_fixation_delay(fixations=fixations, subject=subject)
        preproc_plot.pupil_size_increase(fixations=fixations, subject=subject)
        preproc_plot.performance(subject=subject, bh_data=bh_data)

        print('Plotting scanpaths and trials gaze screens')
        for trial_idx in range(len(bh_data)):
            print(f'\rTrial {trial_idx + 1}', end='')

            preproc_plot.scanpath(fixations=fixations, items_pos=items_pos, bh_data=bh_data, raw=raw,
                                  gazex=meg_gazex_data_clean, gazey=meg_gazey_data_clean, subject=subject, trial_idx=trial_idx)

            preproc_plot.trial_gaze(raw=raw, bh_data=bh_data, gazex=meg_gazex_data_clean, gazey=meg_gazey_data_clean,
                                    subject=subject, trial_idx=trial_idx)

    #---------------- Add scaled data to meg data ----------------#
    preproc_functions.add_et_channels(raw=raw, et_channels_meg=et_channels_meg, et_channel_names=et_channel_names)

    #---------------- Save preprocesed data ----------------#
    save.preprocesed_data(raw=raw, subject=subject, bh_data=bh_data, fixations=fixations, saccades=saccades, config=config)

    # Free up memory
    del(raw)
    del(subject)


# for subject_code in [6, 7, 8, 9, 10, 11, 12, 13]:
for subject_code in range(6, 14):
    preprocess(subject_code=subject_code, plot=True)