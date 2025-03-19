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
        self.subjects_ids = ['15909001', '15912001', '15910001', '15950001', '15911001', '11535009', '16191001',
                             '16200001', '16201001', '09991040', '10925091', '16263002', '16269001', '16960001',
                             '16425003', '16984001', '16990002', '17009001', '17010001', '13229006', '16991002']

        # Subjects bad channels
        self.subjects_bad_channels = {'15909001': ['MLT52', 'MLT53', 'MLT54', 'MLT11', 'MLT21'],
                                      '15912001': [],
                                      '15910001': ['MRT52'],
                                      '15950001': ['MLT53', 'MRT52', 'MRT53'],
                                      '15911001': [],
                                      '11535009': [],
                                      '16191001': ['MRT42', 'MRT51'],
                                      '16200001': [],
                                      '16201001': [],
                                      '16256001': [],
                                      '09991040': ['MRC12', 'MRC13', 'MRC21', 'MRC22', 'MRC51', 'MRC52', 'MRC62', 'MZF01',
                                                   'MZF03', 'MRF33', 'MRF43', 'MRF52', 'MRF53', 'MRF61', 'MRF62'],
                                      '10925091': ['MLT44'],
                                      '16263002': [],
                                      '16269001': ['MLT52'],
                                      '16960001': ['MRF45'],
                                      '16425003': ['MRC17', 'MRC31', 'MRC25', 'MRC54'],
                                      '16984001': ['MRF45', 'MLT52'],
                                      '16990002': ['MRF45', 'MRF25'],
                                      '17009001': ['MLT12', 'MRF45', 'MRF25', 'MLT22', 'MLT23', 'MLT13', 'MLT11'],
                                      '17010001': ['MRF25', 'MRF45'],
                                      '13229006': ['MRF45'],
                                      '16991002': ['MRF45', 'MLT33']
                                      }

        # Distance to the screen during the experiment
        self.screen_distance = {'15909001': 58, '15912001': 58, '15910001': 58, '15950001': 58, '15911001': 58,
                                '11535009': 58, '16191001': 58, '16200001': 58, '16201001': 58, '16256001': 58,
                                '09991040': 58, '10925091': 58, '16263002': 58, '16269001': 58, '16960001': 58,
                                '16425003': 62, '16984001': 59, '16990002': 61, '17009001': 62.5, '17010001': 61,
                                '13229006': 63, '16991002': 66.5}

        # Screen size
        self.screen_size = {'15909001': 38, '15912001': 38, '15910001': 38, '15950001': 38, '15911001': 38,
                            '11535009': 38, '16191001': 38, '16200001': 38, '16201001': 38, '16256001': 38,
                            '09991040': 38, '10925091': 38, '16263002': 38, '16269001': 38, '16960001': 38,
                            '16425003': 38, '16984001': 38, '16990002': 38, '17009001': 38, '17010001': 38,
                            '13229006': 37.5, '16991002': 37.5}

        # Subjects groups
        # For some reason participant 6 has the mapping from balanced participants
        self.subjects_groups = {'15909001': 'Balanced',
                                '15912001': 'Balanced',
                                '15910001': 'Balanced',
                                '15950001': 'Counterbalanced',
                                '15911001': 'Balanced',
                                '11535009': 'Counterbalanced',
                                '16191001': 'Balanced',
                                '16200001': 'Balanced',
                                '16201001': 'Balanced',
                                '16256001': 'Counterbalanced',
                                '09991040': 'Counterbalanced',
                                '10925091': 'Counterbalanced',
                                '16263002': 'Counterbalanced',
                                '16269001': 'Counterbalanced',
                                '16960001': 'Counterbalanced',
                                '16425003': 'Counterbalanced',
                                '16984001': 'Counterbalanced',
                                '16990002': 'Balanced',
                                '17009001': 'Counterbalanced',
                                '17010001': 'Counterbalanced',
                                '13229006': 'Counterbalanced',
                                '16991002': 'Counterbalanced'
                                }

        # Missing triggers subjects
        self.no_trig_subjects = ['15909001', '15912001', '15910001', '15950001', '15911001', '11535009', '16191001',
                                 '16200001', '16201001']

        # Missing bh subjects
        self.missing_bh_subjects = ['16191001', '16200001', '16201001']

        # Define subjects that took the old trials loop experiment
        self.trials_loop_subjects = ['15909001', '15912001']

        # Subjects with no events descriptiion
        self.no_desc_subjects = ['16960001', '16425003', '16984001', '16990002', '17009001', '17010001', '13229006', '16991002']

        # Subjects with no pupil data
        self.no_pupil_subjects = ['16960001', '16425003', '16984001', '16990002', '17009001', '17010001', '13229006', '16991002']

        # Get et channels by name [Gaze x, Gaze y, Pupils]
        self.et_channel_names = ['UADC001-4123', 'UADC002-4123', 'UADC013-4123']

        # Trigger channel name
        self.trig_ch = 'UPPT002'

        # Buttons channel name
        self.button_ch = 'UPPT001'

        # Buttons values to colors map
        self.buttons_ch_map = {1: 'blue', 2: 'yellow', 4: 'green', 8: 'red', 'blue': 1, 'yellow': 2, 'green': 4, 'red': 8}
        self.buttons_pc_map = {1: 'blue', 2: 'yellow', 3: 'green', 4: 'red', 'blue': 1, 'yellow': 2, 'green': 3, 'red': 4}
        # Participant 15950001 was Counterbalance (Correct answers are taken from CB)
        # But the instructions during the experiment were for a Balanced participant
        # We invert the mapping red - blue to compensate this
        self.buttons_pc_map_15950001 = {1: 'red', 2: 'yellow', 3: 'green', 4: 'blue', 'red': 1, 'yellow': 2, 'green': 3, 'blue': 4}

        # DAC delay (in ms)
        self.DAC_delay = 10

        # Background noise recordings date ids
        self.noise_recordings = ['20220530', '20220816', '20220915', '20220916', '20220922', '20220923', '20230623', '20230629', '20230630',
                                 '20230705', '20230706', '20230802']

        # Participants associated background noise
        self.subjects_noise = {'15909001': '20220530',
                               '15912001': '20220530',
                               '15910001': '20220530',
                               '15950001': '20220530',
                               '15911001': '20220530',
                               '11535009': '20220530',
                               '16191001': '20220816',
                               '16200001': '20220816',
                               '16201001': '20220816',
                               '16256001': '20220915',
                               '09991040': '20220916',
                               '10925091': '20220916',
                               '16263002': '20220922',
                               '16269001': '20220923',
                               '16960001': '20230623',
                               '16425003': '20230623',
                               '16984001': '20230629',
                               '16990002': '20230630',
                               '17009001': '20230705',
                               '17010001': '20230706',
                               '13229006': '20230802',
                               '16991002': '20230802'
                               }

        # Notch filter line noise frequencies
        self.background_line_noise_freqs = {'20220530': (50, 100, 110, 150, 200, 250, 300),
                                            '20220816': (50, 100, 110, 150, 200, 250),
                                            '20220915': (50, 100, 110, 150, 200, 250),
                                            '20220916': (50, 100, 110, 150, 200, 250),
                                            '20220922': (50, 100, 110, 150, 200, 250),
                                            '20220923': (50, 100, 110, 150, 200, 250),
                                            '20230623': (50, 57, 100, 109, 150, 200, 250, 300),
                                            '20230629': (50, 57, 100, 109, 150, 200, 250),
                                            '20230630': (50, 57, 100, 109, 150, 200, 250),
                                            '20230705': (50, 57, 100, 109, 150, 200, 250),
                                            '20230706': (50, 57, 100, 109, 150, 200, 250),
                                            '20230802': (50, 57, 100, 109, 150, 200, 250),
                                            '16991002': (50, 57, 100, 109, 150, 200, 250),
                                            }


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
        self.general = self.general()

    class preprocessing:
        def __init__(self):

            # Samples drop at begining of missing pupils signal
            self.start_interval_samples = 24
            # self.start_interval_samples = {'15909001': 12, '15912001': 12, '15910001': 12, '15950001': 12, '15911001': 12,
            #                                '11535009': 12, '16191001': 12, '16200001': 12, '16201001': 12, '16256001': 12,
            #                                '09991040': 12, '10925091': 12, '16263002': 12, '16269001': 12}
            
            # Samples drop at end of missing pupils signal
            self.end_interval_samples = 24
            # self.end_interval_samples = {'15909001': 24, '15912001': 24, '15910001': 24, '15950001': 24, '15911001': 24,
            #                              '11535009': 24, '16191001': 24, '16200001': 24, '16201001': 24, '16256001': 24,
            #                              '09991040': 24, '10925091': 24, '16263002': 24, '16269001': 24}

            # Pupil size threshold to consider missing signal
            self.pupil_thresh = {'15909001': -4.6, '15912001': -4.71, '15910001': -4.113, '15950001': -4.6, '15911001': -4.6,
                                 '11535009': -4.6, '16191001': -4.58, '16200001': -4.56, '16201001': -4.58, '16256001': -4.6,
                                 '09991040': -4.37, '10925091': -4.565, '16263002': -4.39, '16269001': -4.57, '16960001': 300,
                                 '16425003': 200, '16984001': 0, '16990002': 0, '17009001': 0, '17010001': 0,
                                 '13229006': 0, '16991002': 0}

            # Distance to the screen during the experiment
            self.screen_distance = {'15909001': 58, '15912001': 58, '15910001': 58, '15950001': 58, '15911001': 58,
                                    '11535009': 58, '16191001': 58, '16200001': 58, '16201001': 58, '16256001': 58,
                                    '09991040': 58, '10925091': 58, '16263002': 58, '16269001': 58, '16960001': 58,
                                    '16425003': 62, '16984001': 59, '16990002': 61, '17009001': 62.5, '17010001': 61,
                                    '13229006': 63, '16991002': 66.5}

            # Screen size
            self.screen_size = {'15909001': 38, '15912001': 38, '15910001': 38, '15950001': 38, '15911001': 38,
                                '11535009': 38, '16191001': 38, '16200001': 38, '16201001': 38, '16256001': 38,
                                '09991040': 38, '10925091': 38, '16263002': 38, '16269001': 38, '16960001': 38,
                                '16425003': 38, '16984001': 38, '16990002': 38, '17009001': 38, '17010001': 38,
                                '13229006': 37.5, '16991002': 37.5}

            # Et samples shift for ET-MEG alignment
            self.et_samples_shift = {'15909001': {0: 105194, 1: 142301, 2: 178980, 3: 271317, 4: 308180, 5: 346960, 6: 401542},
                                     '15912001': {0: 191406, 1: 240127, 2: 280734, 3: 318607, 4: 374598, 5: 428743, 6: 480315},
                                     '15910001': {0: 201780, 1: 259623, 2: 301018, 3: 320036, 4: 356352, 5: 420185, 6: 434579},
                                     '15950001': {0: 146066, 1: 175241, 2: 193623, 3: 213486, 4: 246432, 5: 261065, 6: 277778},
                                     '15911001': {0: 92835, 1: 115144, 2: 137921, 3: 153215, 4: 174966, 5: 204540, 6: 216924},
                                     '11535009': {0: 68464, 1: 89391, 2: 107763, 3: 145314, 4: 163715, 5: 182900, 6: 197575},
                                     '16191001': {0: 143701, 1: 165824, 2: 207513, 3: 258317, 4: 317730, 5: 371280, 6: 403469},
                                     '16200001': {0: 152662, 1: 186316, 2: 216280, 3: 258707, 4: 348267, 5: 377325, 6: 404956},
                                     '16201001': {0: 96055, 1: 117631, 2: 155282, 3: 198952, 4: 266306, 5: 308630, 6: 331413},
                                     '09991040': {0: 158146, 1: 211028, 2: 282490, 3: 286743, 4: 335281, 5: 391390, 6: 437725},
                                     '10925091': {0: 112968, 1: 164895, 2: 216394, 3: 281356, 4: 332432, 5: 381577, 6: 429920},
                                     '16263002': {0: 144254, 1: 217256, 2: 266348, 3: 315123, 4: 405812, 5: 454113, 6: 504071},
                                     '16269001': {0: 128759, 1: 186451, 2: 191845, 3: 242896, 4: 214075, 5: 223144, 6: 247076},
                                     '16960001': {},
                                     '16425003': {},
                                     '16984001': {},
                                     '16990002': {},
                                     '17009001': {},
                                     '17010001': {},
                                     '13229006': {},
                                     '16991002': {}
                                     }

            # Notch filter line noise frequencies
            self.line_noise_freqs = {'15909001': (50, 100, 110, 150, 200, 250, 300),
                                     '15912001': (50, 100, 110, 150, 200, 250, 300),
                                     '15910001': (50, 100, 110, 150, 200, 250, 300),
                                     '15950001': (50, 100, 110, 150, 200, 250, 300),
                                     '15911001': (50, 100, 110, 150, 200, 250, 300),
                                     '11535009': (50, 100, 110, 150, 200, 250, 300),
                                     '16191001': (50, 100, 110, 150, 200, 250, 300),
                                     '16200001': (50, 100, 110, 150, 200, 250, 300),
                                     '16201001': (50, 100, 110, 150, 200, 250, 300),
                                     '16256001': (50, 100, 110, 150, 200, 250, 300),
                                     '09991040': (50, 100, 110, 150, 200, 250, 300),
                                     '10925091': (50, 100, 110, 150, 200, 250, 300),
                                     '16263002': (50, 100, 110, 150, 200, 250, 300),
                                     '16269001': (50, 100, 110, 150, 200, 250, 300),
                                     '16960001': (50, 57, 100, 109, 150, 200, 250, 300),
                                     '16425003': (50, 57, 100, 109, 150, 200, 250, 300),
                                     '16984001': (50, 57, 100, 109, 150, 200, 250, 300),
                                     '16990002': (50, 57, 100, 109, 150, 200, 250, 300),
                                     '17009001': (50, 57, 100, 109, 150, 200, 250, 300),
                                     '17010001': (50, 57, 100, 109, 150, 200, 250, 300),
                                     '13229006': (50, 57, 100, 109, 150, 200, 250, 300),
                                     '16991002': (50, 57, 100, 109, 150, 200, 250, 300)
                                     }

    class general:
        def __init__(self):
            # Trial reject parameter based on MEG peak to peak amplitude
            self.reject_amp = {'15909001': 1.5e-12, '15912001': 3.5e-12, '15910001': 3e-12, '15950001': 3.5e-12,
                               '15911001': 3.5e-12, '11535009': 3.5e-12, '16191001': 4e-12, '16200001': 2e-12,
                               '16201001': 1.5e-12, '16256001': 3.5e-12, '09991040': 1.2e-12, '10925091': 1.4e-12,
                               '16263002': 2.5e-12, '16269001': 2e-12, '16960001': 5e-12, '16425003': 5e-12,
                               '16984001': 5e-12, '16990002': 5e-12, '17009001': 5e-12, '17010001': 5e-12,
                               '13229006': 5e-12, '16991002': 5e-12}

            # Subjects dev <-> head transformation to use
            self.head_loc_idx = {'15909001': 2, '15912001': 1, '15910001': 3, '15950001': 0, '15911001': 0,
                                 '11535009': 0, '16191001': 2, '16200001': 2, '16201001': 0, '16256001': 0,
                                 '09991040': 0, '10925091': 0, '16263002': 1, '16269001':  2, '16960001': 0,
                                 '16425003': 0, '16984001': 0, '16990002': 0, '17009001': 0, '17010001': 0,
                                 '13229006': 0, '16991002': 0}


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

        # Define subject group and bad channels by matching id index
        self.bad_channels = exp_info.subjects_bad_channels[self.subject_id]
        self.group = exp_info.subjects_groups[self.subject_id]
        self.buttons_pc_map = exp_info.buttons_pc_map
        # Exception for subject 15950001
        if subject_code == '15950001':
            self.buttons_pc_map = exp_info.buttons_pc_map_15950001

        # Get run configuration for subject
        self.config = self.subject_config(config=config, subject_id=self.subject_id)

    # Subject's parameters and configuration
    class subject_config:

        def __init__(self, config, subject_id):
            self.preproc = self.preproc(config=config, subject_id=subject_id)
            self.general = self.general(config=config, subject_id=subject_id)

        # Configuration for preprocessing run
        class preproc:
            def __init__(self, config, subject_id):

                # Get config.preprocessing attributes and get data for corresponding subject
                preproc_attributes = config.preprocessing.__dict__.keys()

                # Iterate over attributes and get data for corresponding subject
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

        # Configuration for further analysis
        class general:
            def __init__(self, config, subject_id):

                # Get config.preprocessing attirbutes and get data for corresponding subject
                general_attributes = config.general.__dict__.keys()

                # Iterate over attributes and get data for conrresponding subject
                for general_att in general_attributes:
                    att = getattr(config.general, general_att)
                    if type(att) == dict:
                        try:
                            # If subject_id in dictionary keys, get attribute, else pass
                            att_value = att[subject_id]
                            setattr(self, general_att, att_value)
                        except:
                            pass
                    else:
                        # If attribute is general for all subjects, get attribute
                        att_value = att
                        setattr(self, general_att, att_value)


    # Raw MEG data
    def load_raw_meg_data(self):
        """
        MEG data for parent subject as Raw instance of MNE.
        """

        print('\nLoading Raw MEG data')
        # Get subject path
        subj_path = pathlib.Path(os.path.join(paths().ctf_path(), self.subject_id))
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

            # Set dev <-> head transformation from optimal head localization
            raw.info['dev_head_t'] = raws_list[config().general.head_loc_idx[self.subject_id]].info['dev_head_t']

        # If only one session return that session as whole raw data
        elif len(ds_files) == 1:
            raw = mne.io.read_raw_ctf(ds_files[0], system_clock='ignore')

        # Missing data
        else:
            raise ValueError('No .ds files found in subject directory: {}'.format(subj_path))

        return raw


    # MEG data
    def load_preproc_meg_data(self, preload=False):
        """
        Preprocessed MEG data for parent subject as raw instance of MNE.
        """

        print('\nLoading Preprocessed MEG data')
        # get subject path
        preproc_path = paths().preproc_path()
        file_path = pathlib.Path(os.path.join(preproc_path, self.subject_id, f'Subject_{self.subject_id}_meg.fif'))

        # Load data
        fif = mne.io.read_raw_fif(file_path, preload=preload)

        return fif


    # ICA MEG data
    def load_ica_meg_data(self, preload=False):
        """
        ICA MEG data for parent subject as Raw instance of MNE.
        """

        print('\nLoading ICA MEG data')
        ica_subj_path = paths().ica_path() + f'{self.subject_id}/'

        # Try to load ica data
        try:
            print(f'Loading ica data for subject {self.subject_id}')
            file_path = pathlib.Path(os.path.join(ica_subj_path, f'Subject_{self.subject_id}_ICA.fif'))
            # Load data
            ica_data = mne.io.read_raw_fif(file_path, preload=preload)

        except:
            raise ValueError(f'No previous ica data found for subject {self.subject_id}')

        return ica_data


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
        subj_path = pathlib.Path(os.path.join(paths().et_path(), self.subject_id))
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
        subj_path = pathlib.Path(os.path.join(paths().bh_path(), self.subject_id))
        bh_file = list(subj_path.glob('*.csv'.format(self.subject_id)))[0]

        # Load DataFrame
        df = pd.read_csv(bh_file)

        return df


class noise:
    """
    Class containing bakground noise data.

    Parameters
    ----------
    exp_info:
    config:
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

    def __init__(self, exp_info, date_id):

        # Noise recording date id
        self.subject_id = date_id

        # Back Noise directory name
        self.bkg_noise_dir = 'BACK_NOISE'

        # Noise data path
        self.ctf_path = pathlib.Path(os.path.join(exp_info.ctf_path, self.bkg_noise_dir))

    # MEG data
    def load_raw_meg_data(self):
        """
        MEG data for parent subject as Raw instance of MNE.
        """

        print('\nLoading MEG data')
        # get subject path
        subj_path = pathlib.Path(os.path.join(paths().ctf_path(), 'BACK_NOISE'))
        ds_files = list(subj_path.glob('QA_*_{}_01.ds'.format(self.subject_id)))
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

        # If only one session return that session as whole raw data
        elif len(ds_files) == 1:
            raw = mne.io.read_raw_ctf(ds_files[0], system_clock='ignore')

        # Missing data
        else:
            raise ValueError(f'No {ds_files} files found in subject directory: {subj_path}')

        return raw


    def load_preproc_data(self, preload=False):
        """
        Preprocessed MEG data for parent subject as raw instance of MNE.
        """

        print('\nLoading Preprocessed MEG data')
        # get subject path
        preproc_path = paths().preproc_path()
        file_path = pathlib.Path(os.path.join(preproc_path, self.bkg_noise_dir, f'{self.subject_id}_meg.fif'))

        # Load data
        fif = mne.io.read_raw_fif(file_path, preload=preload)

        return fif


class all_subjects:

    def __init__(self, all_fixations, all_saccades, all_bh_data, all_rt, all_corr_ans, all_mss):
        self.subject_id = 'All_Subjects'
        self.fixations = all_fixations
        self.saccades = all_saccades
        self.trial = np.arange(1, 211)
        self.bh_data = all_bh_data
        self.rt = all_rt
        self.corr_ans = all_corr_ans
        self.mss = all_mss