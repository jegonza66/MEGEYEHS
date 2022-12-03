import mne
import numpy as np
import functions_general

def define_events(subject, epoch_id, screen, mss, dur, tgt, dir, meg_data, evt_from_df=False, evt_from_annot=True):

    print('Defining events')

    metadata_sup = None

    if evt_from_df:
        if 'fix' in epoch_id:
            metadata = subject.fixations
        elif 'sac' in epoch_id:
            metadata = subject.saccades

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

        metadata.reset_index(drop=True, inplace=True)

        events_samples, event_times = functions_general.find_nearest(meg_data.times, metadata['onset'])

        events = np.zeros((len(events_samples), 3)).astype(int)
        events[:, 0] = events_samples
        events[:, 2] = metadata.index

        events_id = dict(zip(metadata.id, metadata.index))

    if evt_from_annot:
        # Get events from annotations
        all_events, all_event_id = mne.events_from_annotations(meg_data, verbose=False)

        # Select epochs
        epoch_keys = [key for key in all_event_id.keys() if epoch_id in key]
        if screen:
            epoch_keys = [epoch_key for epoch_key in epoch_keys if f'_{screen}_' in epoch_key]
        if mss:
            trials_mss = subject.bh_data.loc[subject.bh_data['Nstim'] == mss].index + 1  # add 1 due to python 0th indexing
            epoch_keys = [epoch_key for epoch_key in epoch_keys for trial_mss in trials_mss if f'_t{trial_mss}_' in epoch_key]

        # Get events and ids matchig selection
        metadata, events, events_id = mne.epochs.make_metadata(events=all_events, event_id=all_event_id,
                                                               row_events=epoch_keys, tmin=0, tmax=0,
                                                               sfreq=meg_data.info['sfreq'])

        if 'fix' in epoch_id:
            metadata_sup = subject.fixations
        elif 'sac' in epoch_id:
            metadata_sup = subject.saccades

    return metadata, events, events_id, metadata_sup