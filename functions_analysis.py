import mne
import numpy as np
import functions_general

def define_events(subject, epoch_id, evt_from_df, evt_from_annot, screen, mss, dur, tgt, dir, meg_data):

    if 'fix' in epoch_id:
        metadata = subject.fixations
    elif 'sac' in epoch_id:
        metadata = subject.saccades

    if evt_from_df:
        # Get events from fix/sac Dataframe
        if screen:
            metadata = metadata.loc[(metadata['screen'] == screen)]
        if mss:
            metadata = metadata.loc[(metadata['mss'] == mss)]
        if dur:
            metadata = metadata.loc[(metadata['duration'] >= dur)]
        if 'fix' in epoch_id:
            if tgt == 1:
                metadata = metadata.loc[(metadata['fix_target'] == tgt)]
            elif tgt == 0:
                metadata = metadata.loc[(metadata['fix_target'] == tgt)]
        if 'sac' in epoch_id:
            if dir:
                metadata = metadata.loc[(metadata['dir'] == dir)]

        events_samples, event_times = functions_general.find_nearest(meg_data.times, metadata['onset'])

        events = np.zeros((len(events_samples), 3)).astype(int)
        events[:, 0] = events_samples
        events[:, 2] = metadata.index

        events_id = dict(zip(metadata.id, metadata.index))

    elif evt_from_annot:
        # Get events from annotations
        all_events, all_event_id = mne.events_from_annotations(meg_data, verbose=False)
        # Select epochs
        # epoch_keys = [key for key in all_event_id.keys() if epoch_id in key]
        trials_mss = subject.bh_data.loc[subject.bh_data['Nstim'] == mss].index + 1  # add 1 due to python 0th indexing
        epoch_keys = [f'{screen}_t{trial_mss}' for trial_mss in trials_mss]

        # Get events and ids matchig selection
        metadata, events, events_id = mne.epochs.make_metadata(events=all_events, event_id=all_event_id,
                                                               row_events=epoch_keys, tmin=0, tmax=0,
                                                               sfreq=meg_data.info['sfreq'])

    return metadata, events, events_id