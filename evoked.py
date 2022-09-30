import load
import os
import mne

from paths import paths

save_path = paths().save_path()
l_freq = 0.5
h_freq = 100

evokeds = []

for subject in [0, 1, 2, 3, 4, 5]:

    subject = load.subject(subject)
    raw = subject.preproc_data()

    # PICK MEG AND STIM CHS
    raw.pick(['meg'])
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
    epoch_id = 'blue'
    epoch_keys = [key for key in event_id.keys() if epoch_id in key]
    epochs = mne.concatenate_epochs([epochs[key] for key in epoch_keys])
    # AVERAGE EPOCHS TO GET EVOKED
    evoked = epochs.average()
    # GET MEG CHS ONLY
    evoked.pick('meg')
    # FILTER
    evoked.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin')
    # Apend to evokeds list to pass to grand average
    evokeds.append(evoked)

    # Save epoched data
    epoch_save_path = save_path + f'Epochs/{epoch_id}_lfreq{l_freq}_hfreq{h_freq}/' + subject.subject_id + '/'
    os.makedirs(epoch_save_path, exist_ok=True)
    epochs_data_fname = f'Subject_{subject.subject_id}_epo.fif'
    epochs.save(epoch_save_path + epochs_data_fname, overwrite=True)

    # Save evoked data
    evoked_save_path = save_path + f'Evoked/{epoch_id}_lfreq{l_freq}_hfreq{h_freq}/' + subject.subject_id + '/'
    os.makedirs(evoked_save_path, exist_ok=True)
    evoked_data_fname = f'Subject_{subject.subject_id}_ave.fif'
    evoked.save(evoked_save_path + evoked_data_fname, overwrite=True)

# Compute grand average
grand_avg = mne.grand_average(evokeds, interpolate_bads=False)
# Save grand average
ga_save_path = save_path + f'Evoked/{epoch_id}_lfreq{l_freq}_hfreq{h_freq}/'
os.makedirs(evoked_save_path, exist_ok=True)
grand_avg_data_fname = f'Grand_average_ave.fif'
evoked.save(ga_save_path + grand_avg_data_fname, overwrite=True)

# PLOT
grand_avg.plot(window_title=f'Grand average - {epoch_id}', gfp=True, time_unit='s')