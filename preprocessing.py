import setup
import load
import save
import plot_preproc
import functions_preproc
from paths import paths


# Load experiment info
exp_info = setup.exp_info()
# Load configuration
config = load.config(path=paths().config_path(), fname='config.pkl')
# Run plots
plot = False

# Run
for subject_code in exp_info.subjects_ids:

    #---------------- Load data ----------------#
    # Define subject
    subject = setup.raw_subject(exp_info=exp_info, config=config, subject_code=subject_code)

    # Load Meg data
    raw = subject.load_raw_meg_data()

    # Get ET channels from MEG
    print('\nGetting ET channels data from MEG')
    et_channels_meg = raw.get_data(picks=exp_info.et_channel_names)

    #---------------- Remove DAC delay samples ----------------#
    meg_gazex_data_raw, meg_gazey_data_raw, meg_pupils_data_raw = \
        functions_preproc.DAC_samples(et_channels_meg=et_channels_meg, exp_info=exp_info, sfreq=raw.info['sfreq'])

    #---------------- Reescaling based on conversion parameters ----------------#
    meg_gazex_data_scaled, meg_gazey_data_scaled = functions_preproc.reescale_et_channels(meg_gazex_data_raw=meg_gazex_data_raw,
                                                                                          meg_gazey_data_raw=meg_gazey_data_raw)

    #---------------- Blinks removal ----------------#
    # Define intervals around blinks to also fill with nan. Due to conversion noise from square signal
    et_channels_meg = functions_preproc.blinks_to_nan(meg_gazex_data_scaled=meg_gazex_data_scaled, meg_gazey_data_scaled=meg_gazey_data_scaled,
                                                      meg_pupils_data_raw=meg_pupils_data_raw, config=subject.config.preproc)

    #---------------- Defining response events and trials ----------------#
    if subject.subject_id in exp_info.no_trig_subjects:
        raw, subject = functions_preproc.define_events_trials_ET(raw=raw, subject=subject, config=config, exp_info=exp_info)
    else:
        raw, subject = functions_preproc.define_events_trials_trig(raw=raw, subject=subject, config=config, exp_info=exp_info)

    #---------------- Fixations and saccades detection ----------------#
    fixations, saccades, subject = functions_preproc.fixations_saccades_detection(raw=raw, et_channels_meg=et_channels_meg,
                                                                                  subject=subject)

    # ---------------- Saccades classification ----------------#
    saccades, raw, subject = functions_preproc.saccades_classification(subject=subject, saccades=saccades, raw=raw)

    #---------------- Fixations classification ----------------#
    fixations, raw = functions_preproc.fixation_classification(subject=subject, fixations=fixations, raw=raw)

    #---------------- Items classification ----------------#
    raw, subject, items_pos = functions_preproc.target_vs_distractor(fixations=fixations, subject=subject,
                                                                     raw=raw, distance_threshold=70)

    #---------------- Save fix time distribution, pupils size vs mss, scanpath and trial gaze figures ----------------#
    if plot:
        plot_preproc.first_fixation_delay(subject=subject)
        plot_preproc.pupil_size_increase(subject=subject)
        plot_preproc.performance(subject=subject)
        plot_preproc.fixation_duration(subject=subject)
        plot_preproc.saccades_amplitude(subject=subject)
        plot_preproc.saccades_dir_hist(subject=subject)
        plot_preproc.sac_main_seq(subject=subject)

        print('Plotting scanpaths and trials gaze screens')
        for trial_idx in range(len(subject.bh_data)):
            print(f'\rTrial {trial_idx + 1}', end='')

            plot_preproc.scanpath(raw=raw, subject=subject, items_pos=items_pos, et_channels_meg=et_channels_meg, trial_idx=trial_idx)

            plot_preproc.trial_gaze(raw=raw, subject=subject, et_channels_meg=et_channels_meg, trial_idx=trial_idx)

        for block_num in range(len(subject.emap)):
            plot_preproc.emap_gaze(raw=raw, subject=subject, et_channels_meg=et_channels_meg, block_num=block_num)

    #---------------- Add scaled data to meg data ----------------#
    functions_preproc.add_et_channels(raw=raw, et_channels_meg=et_channels_meg, et_channel_names=exp_info.et_channel_names)

    #---------------- Save preprocesed data ----------------#
    save.preprocessed_data(raw=raw, et_data_scaled=et_channels_meg, subject=subject, config=config)

    # Free up memory
    del(raw)
    del(subject)