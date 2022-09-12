from paths import paths

import mne
import os
import pandas as pd
import pathlib
import numpy as np


class subject:
    """
    Class containing subjects data.

    Parameters
    ----------
    subject: {'int', 'str'}, default=None
        Subject id (str) or number (int). If None, takes the first subject.

    Attributes
    -------
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

    def __init__(self, subject=None):
        # Define ctf data path and files path
        self.ctf_path = paths().ctf_path()
        self.et_path = paths().et_path()
        self.beh_path = paths().beh_path()
        self.mri_path = paths().mri_path()
        self.opt_path = paths().opt_path()

        # Select subject
        subjects_ids = ['15909001', '15912001', '15910001', '15950001', '15911001', '11535009', 'BACK_NOISE']
        # Subjects bad channels
        subjects_bad_channels = [['MLT11-4123', 'MLT21-4123'], [], [], [], [], [], []]
        # Subjects groups
        subjects_groups = ['Balanced', 'Balanced', 'Balanced', 'Counter-balanced', 'Balanced', 'Counter-balanced' 'X']

        # Select 1st subject by default
        if subject == None:
            self.subject_id = subjects_ids[0]
            self.bad_channels = subjects_bad_channels[0]
            self.group = subjects_groups[0]
        # Select subject by index
        elif type(subject) == int:
            self.subject_id = subjects_ids[subject]
            self.bad_channels = subjects_bad_channels[subject]
            self.group = subjects_groups[subject]
        # Select subject by id
        elif type(subject) == str and (subject in subjects_ids):
            self.subject_id = subject
            self.bad_channels = subjects_bad_channels[subjects_ids.index(subject)]
            self.group = subjects_groups[subjects_ids.index(subject)]
        else:
            print('Subject not found.')

        # Mapping between button value and color
        if self.group == 'Balanced':
            self.map = {'blue': '1', 'red': '4'}
        elif self.group == 'Counter-balanced':
            self.map = {'blue': '4', 'red': '1'}


    # MEG data
    def ctf_data(self):
        """
        MEG data for parent subject as Raw instance of MNE.
        """

        print('\nLoading MEG data')
        # get subject path
        subj_path = pathlib.Path(os.path.join(self.ctf_path, self.subject_id))
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
    def et_data(self):
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
        subj_path = pathlib.Path(os.path.join(self.et_path, self.subject_id))
        asc_file_path = list(subj_path.glob('*{}.asc'.format(self.subject_id)))[0]

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
    def beh_data(self):
        """
        Behavioural data for parent subject as pandas DataFrames.
        """
        # Get subject path
        subj_path = pathlib.Path(os.path.join(self.beh_path, self.subject_id))
        beh_file = list(subj_path.glob('*{}*[!ORIGINAL].csv'.format(self.subject_id)))[0]

        # Load DataFrame
        df = pd.read_csv(beh_file)

        return df
