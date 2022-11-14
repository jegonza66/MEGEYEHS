import setup
import load
import save
import preproc_plot
import preproc_functions
from paths import paths


def preprocess(subject_code, exp_info, config, plot=False):
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
        preproc_functions.DAC_samples(et_channels_meg=et_channels_meg, exp_info=exp_info, sfreq=raw.info['sfreq'])

    #---------------- Reescaling based on conversion parameters ----------------#
    meg_gazex_data_scaled, meg_gazey_data_scaled = preproc_functions.reescale_et_channels(meg_gazex_data_raw=meg_gazex_data_raw,
                                                                                          meg_gazey_data_raw=meg_gazey_data_raw)

    #---------------- Blinks removal ----------------#
    # Define intervals around blinks to also fill with nan. Due to conversion noise from square signal
    et_channels_meg = preproc_functions.blinks_to_nan(meg_gazex_data_scaled=meg_gazex_data_scaled, meg_gazey_data_scaled=meg_gazey_data_scaled,
                                                      meg_pupils_data_raw=meg_pupils_data_raw, config=subject.config.preproc)

    #---------------- Defining response events and trials ----------------#
    if subject.subject_id in exp_info.no_trig_subjects:
        raw, subject = preproc_functions.define_events_trials_ET(raw=raw, subject=subject, config=config, exp_info=exp_info)
    else:
        raw, subject = preproc_functions.define_events_trials_trig(raw=raw, subject=subject, config=config, exp_info=exp_info)

    #---------------- Fixations and saccades detection ----------------#
    fixations, saccades, subject = preproc_functions.fixations_saccades_detection(raw=raw, et_channels_meg=et_channels_meg,
                                                                                  subject=subject, force_run=True)

    # ---------------- Saccades classification ----------------#
    saccades, raw, subject = preproc_functions.saccades_classification(subject=subject, saccades=saccades, raw=raw)

    #---------------- Fixations classification ----------------#
    fixations, raw = preproc_functions.fixation_classification(subject=subject, fixations=fixations, raw=raw)

    #---------------- Items classification ----------------#
    raw, subject, items_pos = preproc_functions.target_vs_distractor(fixations=fixations, subject=subject,
                                                                     raw=raw, distance_threshold=70)

    #---------------- Save fix time distribution, pupils size vs mss, scanpath and trial gaze figures ----------------#
    if plot:
        preproc_plot.first_fixation_delay(subject=subject)
        preproc_plot.pupil_size_increase(subject=subject)
        preproc_plot.performance(subject=subject)
        preproc_plot.fixation_duration(subject=subject)
        preproc_plot.saccades_amplitude(subject=subject)
        preproc_plot.saccades_dir_hist(subject=subject)
        preproc_plot.sac_main_seq(subject=subject)

        print('Plotting scanpaths and trials gaze screens')
        for trial_idx in range(len(subject.bh_data)):
            print(f'\rTrial {trial_idx + 1}', end='')

            preproc_plot.scanpath(raw=raw, subject=subject, items_pos=items_pos, et_channels_meg=et_channels_meg, trial_idx=trial_idx)

            preproc_plot.trial_gaze(raw=raw, subject=subject, et_channels_meg=et_channels_meg, trial_idx=trial_idx)

        for block_num in range(len(subject.emap)):
            preproc_plot.emap_gaze(raw=raw, subject=subject, et_channels_meg=et_channels_meg, block_num=block_num)

    #---------------- Add scaled data to meg data ----------------#
    preproc_functions.add_et_channels(raw=raw, et_channels_meg=et_channels_meg, et_channel_names=exp_info.et_channel_names)

    #---------------- Save preprocesed data ----------------#
    save.preprocessed_data(raw=raw, subject=subject, config=config)

    # Free up memory
    del(raw)
    del(subject)


# Run
# Load experiment info
exp_info = setup.exp_info()
# Load configuration
config = load.config(path=paths().config_path(), fname='config.pkl')

for subject_code in exp_info.subjects_ids:
    preprocess(subject_code=subject_code, exp_info=exp_info, config=config, plot=True)