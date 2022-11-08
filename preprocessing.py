import setup
import load
import save
import preproc_plot
import preproc_functions
from paths import paths

import matplotlib.pyplot as plt
from scipy import signal as sgn

def preprocess(subject_code, exp_info, config, plot=False):
    #---------------- Load data ----------------#
    # Define subject
    config = load.config(path=paths().config_path(), fname='config.pkl')
    subject = setup.raw_subject(exp_info=exp_info, config=config, subject_code=subject_code)

    # Load Meg data
    raw = subject.load_raw_meg_data()

    print('\nGetting ET channels data from MEG')
    et_channels_meg = raw.get_data(picks=exp_info.et_channel_names)

    #---------------- Remove DAC delay samples ----------------#
    meg_gazex_data_raw, meg_gazey_data_raw, meg_pupils_data_raw = \
        preproc_functions.DAC_samples(et_channels_meg=et_channels_meg, exp_info=exp_info, sfreq=raw.info['sfreq'])

    #---------------- Reescaling based on conversion parameters ----------------#
    meg_gazex_data_scaled, meg_gazey_data_scaled = preproc_functions.reescale_et_channels(meg_gazex_data_raw=meg_gazex_data_raw,
                                                                                          meg_gazey_data_raw=meg_gazey_data_raw)

    #---------------- Blinks removal ----------------#
    # Define intervals around blinks to also fill with nan. Due to conversion noise from square signal
    meg_gazex_data_clean, meg_gazey_data_clean, meg_pupils_data_clean = preproc_functions.blinks_to_nan(meg_gazex_data_scaled=meg_gazex_data_scaled,
                                                                                                        meg_gazey_data_scaled=meg_gazey_data_scaled,
                                                                                                        meg_pupils_data_raw=meg_pupils_data_raw,
                                                                                                        config=subject.config.preproc)


    peaks = sgn.find_peaks(-meg_pupils_data_clean, prominence=0.15, width=[0, 160])
    plt.figure()
    plt.plot(meg_pupils_data_raw)
    plt.plot(meg_pupils_data_clean)
    plt.plot(peaks[0], meg_pupils_data_clean[peaks[0]], '.')

    #---------------- Missing signal interpolation ----------------#
    # et_channels_meg = preproc_functions.fake_blink_interpolate(meg_gazex_data_clean=meg_gazex_data_clean,
    #                                                            meg_gazey_data_clean=meg_gazey_data_clean,
    #                                                            meg_pupils_data_clean=meg_pupils_data_clean,
    #                                                            config=config.preprocessing, sfreq=raw.info['sfreq'])

    #---------------- Defining response events and trials ----------------#
    if subject.subject_id in exp_info.no_trig_subjects:
        raw, subject = preproc_functions.define_events_trials_ET(raw=raw, subject=subject, config=config, exp_info=exp_info)
    else:
        raw, subject = preproc_functions.define_events_trials_trig(raw=raw, subject=subject, config=config, exp_info=exp_info)

    #---------------- Fixations and saccades detection ----------------#
    fixations, saccades = preproc_functions.fixations_saccades_detection(raw=raw, meg_gazex_data_clean=meg_gazex_data_clean,
                                                                         meg_gazey_data_clean=meg_gazey_data_clean,
                                                                         subject=subject)

    # ---------------- Saccades classification ----------------#
    saccades, raw, subject = preproc_functions.saccades_classification(subject=subject, saccades=saccades, raw=raw)

    #---------------- Fixations classification ----------------#
    fixations, raw = preproc_functions.fixation_classification(subject=subject, fixations=fixations, saccades=saccades,
                                                               raw=raw, meg_pupils_data_clean=meg_pupils_data_clean)

    #---------------- Items classification ----------------#
    raw, subject, items_pos = preproc_functions.target_vs_distractor(fixations=fixations, subject=subject,
                                                                     raw=raw, distance_threshold=100)

    #---------------- Save fix time distribution, pupils size vs mss, scanpath and trial gaze figures ----------------#
    if plot:
        preproc_plot.first_fixation_delay(subject=subject)
        preproc_plot.pupil_size_increase(subject=subject)
        preproc_plot.performance(subject=subject)

        print('Plotting scanpaths and trials gaze screens')
        for trial_idx in range(len(subject.bh_data)):
            print(f'\rTrial {trial_idx + 1}', end='')

            preproc_plot.scanpath(raw=raw, subject=subject, items_pos=items_pos, gaze_x=meg_gazex_data_clean,
                                  gaze_y=meg_gazey_data_clean, trial_idx=trial_idx)

            preproc_plot.trial_gaze(raw=raw, subject=subject, gaze_x=meg_gazex_data_clean, gaze_y=meg_gazey_data_clean,
                                    trial_idx=trial_idx)

        for block_num in range(len(subject.emap)):
            preproc_plot.emap_gaze(raw=raw, subject=subject, gaze_x=meg_gazex_data_clean, gaze_y=meg_gazey_data_clean,
                                   block_num=block_num)

    #---------------- Add scaled data to meg data ----------------#
    preproc_functions.add_et_channels(raw=raw, et_channels_meg=et_channels_meg, et_channel_names=exp_info.et_channel_names)

    #---------------- Save preprocesed data ----------------#
    save.preprocesed_data(raw=raw, subject=subject, config=config)

    # Free up memory
    del(raw)
    del(subject)


# Run
# Load experiment info
exp_info = setup.exp_info()
# Load configuration
config = load.config(path=paths().config_path(), fname='config.pkl')

# for subject_code in [6, 7, 8, 9, 10, 11, 12]:
for subject_code in exp_info.subjects_ids:
    preprocess(subject_code=subject_code, exp_info=exp_info, config=config, plot=False)