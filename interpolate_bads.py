import functions_preproc
import setup
import load
from paths import paths

# Load experiment info
exp_info = setup.exp_info()

# Digitalization data path
dig_path = paths().opt_path()

# Preprocessed data save path
preproc_data_path = paths().preproc_path()

# for subject_code in exp_info.subjects_ids:
for subject_code in ['09991040']:
    # Load subject and meg preprocessed data
    subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)
    meg_data = subject.load_preproc_meg_data()

    # Set digitalization info in meg_data
    meg_data = functions_preproc.set_digitlization(subject=subject, meg_data=meg_data)

    # Load data to run interpolation
    meg_data.load_data()

    # Interpolate bad channels
    meg_data.interpolate_bads()

    # Save as preprocessed
    preproc_save_path = preproc_data_path + subject.subject_id + '/'
    preproc_meg_data_fname = f'Subject_{subject.subject_id}_meg.fif'
    meg_data.save(preproc_save_path + preproc_meg_data_fname, overwrite=True)


