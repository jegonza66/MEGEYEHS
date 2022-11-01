from paths import paths

import mne
import pandas as pd
import numpy as np
import pathlib
import os


class exp_info:
    """
    Class containing the experiment information.

    Attributes
    -------
    beh_path: str
        Path to the behavioural data.
    ctf_path: str
        Path to the MEG data.
    et_path: str
        Path to the Eye-Tracker data.
    mri_path: str
        Path to the MRI data.
    opt_path: str
        Path to the Digitalization data.
    subjects_ids: list
        List of subject's id.
    subjects_bad_channels: list
        List of subject's bad channels.
    subjects_groups: list
        List of subject's group
    missing_bh_subjects: list
        List of subject's ids missing behavioural data.
    trials_loop_subjects: list
        List of subject;s ids for subjects that took the firts version of the experiment.
    """

    def __init__(self):
    # Define ctf data path and files path
        self.ctf_path = paths().ctf_path()
        self.et_path = paths().et_path()
        self.bh_path = paths().bh_path()
        self.mri_path = paths().mri_path()
        self.opt_path = paths().opt_path()

        # Select subject
        self.subjects_ids = ['15909001', '15912001', '15910001', '15950001', '15911001', '11535009', '16191001', '16200001',
                             '16201001', '09991040', '10925091', '16263002', '16269001', 'BACK_NOISE', '16256001']

        # Subjects bad channels
        self.subjects_bad_channels = {'15909001': ['MLT11', 'MLT21'], '15912001': ['MRT53-4123'],
                                      '15910001': [], '15950001': [], '15911001': [], '11535009': [],
                                      '16191001': ['MRT53-4123'], '16200001': [], '16201001': [], '16256001': [],
                                      '09991040': [], '10925091': [], '16263002': [], '16269001': []}
        # Subjects groups
        # For some reason participant 6 has the mapping from balanced participants
        self.subjects_groups = {'15909001': 'Balanced', '15912001': 'Balanced', '15910001': 'Balanced',
                                '15950001': 'Counterbalanced', '15911001': 'Balanced', '11535009': 'Balanced',
                                '16191001': 'Balanced', '16200001': 'Balanced', '16201001': 'Balanced',
                                '16256001': 'Counterbalanced', '09991040': 'Balanced',
                                '10925091': 'Balanced', '16263002':  'Balanced', '16269001':  'Balanced'}

        # Missing bh subjects
        self.no_trig_subjects = ['15909001', '15912001', '15910001', '15950001', '15911001', '11535009', '16191001',
                                 '16200001', '16201001']

        # Missing bh subjects
        self.missing_bh_subjects = ['16191001', '16200001', '16201001']

        # Define subjects that took the old trials loop experiment
        self.trials_loop_subjects = ['15909001', '15912001']

        # Get et channels by name [Gaze x, Gaze y, Pupils]
        self.et_channel_names = ['UADC001-4123', 'UADC002-4123', 'UADC013-4123']

        # Trigger channel name
        self.trig_ch = 'UPPT002'

        # DAC delay (in ms)
        self.DAC_delay = 8


class config:
    """
    Class containing the run configuration.

    Attributes
    -------
    update_config: bool
        Whether to update/save the configuration or not.
    ctf_path: str
        Path to the MEG data.
    et_path: str
        Path to the Eye-Tracker data.
    mri_path: str
        Path to the MRI data.
    opt_path: str
        Path to the Digitalization data.
    subjects_ids: list
        List of subject's id.
    subjects_bad_channels: list
        List of subject's bad channels.
    subjects_groups: list
        List of subject's group
    missing_bh_subjects: list
        List of subject's ids missing behavioural data.
    trials_loop_subjects: list
        List of subject;s ids for subjects that took the firts version of the experiment.
    """

    def __init__(self):
        self.update_config = True
        self.preprocessing = self.preprocessing()

    class preprocessing:
        def __init__(self):
            # Pupil size threshold to consider missing signal
            self.pupil_thresh = {'15909001': -4.6, '15912001': -4.6, '15910001': -4.6, '15950001': -4.6, '15911001': -4.6,
                                 '11535009': -4.6, '16191001': -4.6, '16200001': -4.6, '16201001': -4.6, '16256001': -4.6,
                                 '09991040': -4.6, '10925091': -4.6, '16263002': -4.6, '16269001': -4.6}

            # Distance to the screen durin the experiment
            self.screen_distance = {'15909001': 58, '15912001': 58, '15910001': 58, '15950001': 58,
                                    '15911001': 58, '11535009': 58, '16191001': 58, '16200001': 58,
                                    '16201001': 58, '16256001': 58, '09991040': 58, '10925091': 58,
                                    '16263002': 58, '16269001': 58}

            self.blink_min_dur = 70
            self.start_interval_samples = 12
            self.end_interval_samples = 24


class raw_subject:
    """
    Class containing subjects data.

    Parameters
    ----------
    subject: {'int', 'str'}, default=None
        Subject id (str) or number (int). If None, takes the first subject.

    Attributes
    ----------
    bad_channels: list
        List of bad channels.
    beh_path: str
        Path to the behavioural data.
    ctf_path: str
        Path to the MEG data.
    et_path: str
        Path to the Eye-Tracker data.
    mri_path: str
        Path to the MRI data.
    opt_path: str
        Path to the Digitalization data.
    subject_id: str
        Subject id.
    """

    def __init__(self, exp_info, config, subject_code=None):

        # Select 1st subject by default
        if subject_code == None:
            self.subject_id = exp_info.subjects_ids[0]
        # Select subject by index
        elif type(subject_code) == int:
            self.subject_id = exp_info.subjects_ids[subject_code]
        # Select subject by id
        elif type(subject_code) == str and (subject_code in exp_info.subjects_ids):
            self.subject_id = subject_code
        else:
            print('Subject not found')

        # Subject's data paths
        self.ctf_path = pathlib.Path(os.path.join(exp_info.ctf_path, self.subject_id))
        self.et_path = pathlib.Path(os.path.join(exp_info.et_path, self.subject_id))
        self.bh_path = pathlib.Path(os.path.join(exp_info.bh_path, self.subject_id))
        self.mri_path = pathlib.Path(os.path.join(exp_info.mri_path, self.subject_id))
        self.opt_path = pathlib.Path(os.path.join(exp_info.opt_path, self.subject_id))

        # Define subject group and bad channels by matching id index
        self.bad_channels = exp_info.subjects_bad_channels[self.subject_id]
        self.group = exp_info.subjects_groups[self.subject_id]

        # Define mapping between button value and color by group
        if self.group == 'Balanced':
            self.map = {'blue': '1', 'red': '4'}
        elif self.group == 'Counterbalanced':
            self.map = {'blue': '4', 'red': '1'}

        # Get run configuration for subject
        self.config = self.subject_config(exp_info=exp_info, config=config, subject_id=self.subject_id)


    # Subject's run configuration
    class subject_config:

        def __init__(self, exp_info, config, subject_id):
            self.preproc = self.preproc(exp_info=exp_info, config=config, subject_id=subject_id)

        class preproc:
            def __init__(self, exp_info, config, subject_id):

                # Get config.preprocessing attirbutes and get data for corresponding subject
                preproc_attributes = config.preprocessing.__dict__.keys()

                # Iterate over attributes and get data for conrresponding subject
                for preproc_att in preproc_attributes:
                    att = getattr(config.preprocessing, preproc_att)
                    if type(att) == dict:
                        try:
                            # If subject_id in dictionary keys, get attribute, else pass
                            att_value = att[subject_id]
                            setattr(self, preproc_att, att_value)
                        except:
                            pass
                    else:
                        # If attribute is general for all subjects, get attribute
                        att_value = att
                        setattr(self, preproc_att, att_value)


    # MEG data
    def load_raw_meg_data(self):
        """
        MEG data for parent subject as Raw instance of MNE.
        """

        print('\nLoading MEG data')
        # get subject path
        subj_path = self.ctf_path
        ds_files = list(subj_path.glob('*{}*.ds'.format(self.subject_id)))
        ds_files.sort()

        # Load sesions
        # If more than 1 session concatenate all data to one raw data
        if len(ds_files) > 1:
            raws_list = []
            for i in range(len(ds_files)):
                raw = mne.io.read_raw_ctf(ds_files[i], system_clock='ignore')
                raws_list.append(raw)
            # MEG data structure
            raw = mne.io.concatenate_raws(raws_list, on_mismatch='ignore')
            return raw
        # If only one session return that session as whole raw data
        elif len(ds_files) == 1:
            raw = mne.io.read_raw_ctf(ds_files[0], system_clock='ignore')
            return raw
        # Missing data
        else:
            print('No .ds files found in subject directory: {}'.format(subj_path))


    # ET data
    def load_raw_et_data(self):
        """
        Eye-Tracker data for parent subject as dict containing pandas DataFrames.

        Attributes
        -------
        asc: DataFrame
            Entire asc file data.
        start_time: str
            Recording start time in ms.
        samples_start: int
            Line number in asc file where ET samples start.
        head: DataFrame
            asc file header.
        eye: str
            Tracked eye.
        samples: DataFrame
            ET data. Columns: Time, Gaze x, Gaze y, Pupil size
        time: Series
            Data time series. Corresponding to the first columns of the samples DataFrame.
        fix: DataFrame
            Fixations.
        sac: DataFrame:
            Saccades.
        blinks: DataFrame
            Blinks.
        sync: DataFrame
            Synchronization messages.
        msg: DataFrame
            All Messages recieved by the Eye-Tracker.
        calibration: Series
            Calibration messages. First value indicates time of message recieved.
        """

        print('\nLoading ET data')
        # get subject path
        subj_path = self.et_path
        # Load asc file
        asc_file_path = list(subj_path.glob('*{}*.asc'.format(self.subject_id)))[0]

        # data structure
        et = {}

        # ASC FILE
        print('Reading asc')
        et['asc'] = pd.read_table(asc_file_path, names=np.arange(9), low_memory=False)

        # INFO
        et['start_time'] = et['asc'][1][np.where(et['asc'][0] == 'START')[0][0]]
        et['samples_start'] = np.where(et['asc'][0] == et['start_time'].split()[0])[0][0]
        et['head'] = et['asc'].iloc[:et['samples_start']]
        et['eye'] = et['head'].loc[et['head'][0] == 'EVENTS'][2].values[0]
        print('Loading headers: {}'.format(et['samples_start']))

        # auxiliar numeric df
        print('Converting asc to numeric')
        num_et = et['asc'].apply(pd.to_numeric, errors='coerce')

        # SAMPLES
        # et['samples'] = num_et.loc[~pd.isna(num_et[np.arange(4)]).any(1)][np.arange(4)]
        et['samples'] = num_et.loc[~pd.isna(num_et[0])][np.arange(4)]
        print('Loading samples: {}'.format(len(et['samples'])))

        # TIME
        et['time'] = num_et[0].loc[~pd.isna(num_et[0])]
        # et['time'] = num_et[0]
        print('Loading time: {}'.format(len(et['time'])))

        # FIXATIONS
        et['fix'] = et['asc'].loc[et['asc'][0].str.contains('EFIX').values == True][np.arange(5)]
        et['fix'][0] = et['fix'][0].str.split().str[-1]
        et['fix'] = et['fix'].apply(pd.to_numeric, errors='coerce')
        print('Loading fixations: {}'.format(len(et['fix'])))

        # SACADES
        et['sac'] = et['asc'].loc[et['asc'][0].str.contains('ESAC').values == True][np.arange(9)]
        et['sac'][0] = et['sac'][0].str.split().str[-1]
        et['sac'] = et['sac'].apply(pd.to_numeric, errors='coerce')
        print('Loading saccades: {}'.format(len(et['sac'])))

        # BLINKS
        et['blinks'] = et['asc'].loc[et['asc'][0].str.contains('EBLINK').values == True][np.arange(3)]
        et['blinks'][0] = et['blinks'][0].str.split().str[-1]
        et['blinks'] = et['blinks'].apply(pd.to_numeric, errors='coerce')
        print('Loading blinks: {}'.format(len(et['blinks'])))

        # ETSYNC
        et['sync'] = et['asc'].loc[et['asc'][1].str.contains('ETSYNC').values == True][np.arange(3)]
        et['sync'][0] = et['sync'][1].str.split().str[1]
        et['sync'][2] = pd.to_numeric(et['sync'][1].str.split().str[2])
        et['sync'][1] = pd.to_numeric(et['sync'][1].str.split().str[0])
        print('Loading sync messages: {}'.format(len(et['sync'])))

        # MESSAGES
        et['msg'] = et['asc'].loc[et['asc'][0].str.contains('MSG').values == True][np.arange(2)]
        print('Loading all messages: {}'.format(len(et['msg'])))

        # CALIBRATION
        et['calibration'] = et['asc'].loc[et['asc'][0].str.contains('CALIBRATION').values == True][0]
        print('Loading calibration messages: {}'.format(len(et['calibration'])))

        return et


    # Behavioural data
    def load_raw_bh_data(self):
        """
        Behavioural data for parent subject as pandas DataFrames.
        """
        # Get subject path
        subj_path = self.bh_path
        bh_file = list(subj_path.glob('*.csv'.format(self.subject_id)))[0]

        # Load DataFrame
        df = pd.read_csv(bh_file)

        return df


    # MEG data
    def load_preproc_meg(self):
        """
        Preprocessed MEG data for parent subject as raw instance of MNE.
        """

        print('\nLoading Preprocessed MEG data')
        # get subject path
        preproc_path = paths().preproc_path()
        file_path = pathlib.Path(os.path.join(preproc_path, self.subject_id, f'Subject_{self.subject_id}_meg.fif'))

        # Load data
        fif = mne.io.read_raw_fif(file_path)

        return fif