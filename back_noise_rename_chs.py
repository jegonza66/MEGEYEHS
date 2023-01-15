# Preprocess background noise (rename channels)
import os
import functions_general
import setup
from paths import paths

exp_info = setup.exp_info()

noise = setup.noise(exp_info=exp_info, id='BACK_NOISE')
raw_noise = noise.load_raw_meg_data()
raw_noise.pick('meg')
raw_noise.rename_channels(functions_general.ch_name_map)

preproc_data_path = paths().preproc_path()
preproc_save_path = preproc_data_path + noise.id + '/'
os.makedirs(preproc_save_path, exist_ok=True)

preproc_meg_data_fname = f'{noise.id}_meg.fif'
raw_noise.save(preproc_save_path + preproc_meg_data_fname, overwrite=True)