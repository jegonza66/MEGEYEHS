import load
import mne

subject = load.subject()
raw = subject.preproc_data()

# PICK MEG AND STIM CHS
raw.pick(['meg', 'misc'])
# Exclude bad channels
bads = subject.bad_channels
raw.info['bads'].extend(bads)

# EPOCH DATA BASED ON BUTTON BOX
reject = dict(mag=4e-12)

# Get events from annotations
events, event_id = mne.events_from_annotations(raw)

# Epoch data
epochs = mne.Epochs(raw, events, event_id=event_id, reject=reject, event_repeated='merge')

# Select epochs
epoch_id = 'fix_ms'
epoch_keys = [key for key in event_id.keys() if epoch_id in key]
epochs_standard = mne.concatenate_epochs([epochs[key] for key in epoch_keys])
# AVERAGE EPOCHS TO GET EVOKED
evoked_std = epochs_standard.average()
# GET MEG CHS ONLY
evoked_std.pick('meg')
# FILTER
evoked_std.filter(l_freq=0.5, h_freq=80., fir_design='firwin')
# PLOT
evoked_std.plot(window_title='Standard', gfp=True, time_unit='s')