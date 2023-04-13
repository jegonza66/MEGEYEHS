# Preprocess background noise
import os
import functions_general
import setup
from paths import paths
import mne
import plot_preproc

exp_info = setup.exp_info()

for noise_date_id in exp_info.noise_recordings:

    # Define background noise recording session
    noise = setup.noise(exp_info=exp_info, date_id=noise_date_id)

    # Load noise data
    raw_noise = noise.load_raw_meg_data()
    raw_noise.load_data()
    # Get only meg channels
    raw_noise.pick('meg')

    # Crop first noise file to 60 seconds to avoid recording end noise
    if noise_date_id == exp_info.noise_recordings[0]:
        raw_noise.crop(tmax=60)

    # Rename channels to match naming convention from data
    raw_noise.rename_channels(functions_general.ch_name_map)

    # Pick filter channels
    meg_picks = mne.pick_types(raw_noise.info, meg=True)

    # Filter
    filtered_data = raw_noise.copy().notch_filter(freqs=exp_info.line_noise_freqs[:-1], picks=meg_picks)

    # Plot
    fig_path = paths().plots_path() + 'Preprocessing/' + f'{noise.bkg_noise_dir}/'
    fig_name = noise.subject_id + 'RAW-Filtered_PSD'
    plot_preproc.line_noise_psd(subject=noise, raw=raw_noise, filtered=filtered_data, display_fig=False,
                                save_fig=True, fig_path=fig_path, fig_name=fig_name)

    # 3rd order gradient compensation
    if filtered_data.compensation_grade != 3:
        filtered_data.apply_gradient_compensation(grade=3, verbose=None)

    # Save
    preproc_data_path = paths().preproc_path()
    preproc_save_path = preproc_data_path + f'{noise.bkg_noise_dir}/'
    os.makedirs(preproc_save_path, exist_ok=True)

    preproc_meg_data_fname = f'{noise.subject_id}_meg.fif'
    filtered_data.save(preproc_save_path + preproc_meg_data_fname, overwrite=True)