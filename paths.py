import os

class paths:
    """
    Paths to participants data.

    Attributes
    -------
    name: str
        Runing computers name result of os.popen('whoami').read().
    """

    def __init__(self):
        self.name = os.popen('whoami').read()


    def ctf_path(self):
        """
        Paths to participants MEG data.

        Returns
        -------
        ctf_path: str
            Path in str format to the CTF folder containing the MEG data.
        """

        if self.name == 'laptop-5i5qsv76\\joaco\n': # Asus Rog
            ctf_path = 'C:/Users/joaco/OneDrive - The University of Nottingham/MEGEYEHS/DATA/CTF_DATA/'
        elif self.name == 'usuario\n': # Liaa Colores
            ctf_path = '/mnt/6a6fd40a-e256-4844-8004-0e60d95969e8/MEGEYEHS/DATA/CTF_DATA/'
        else: # Notts
            ctf_path = 'C:/Users/lpajg1/OneDrive - The University of Nottingham/MEGEYEHS/DATA/CTF_DATA/'
        return ctf_path


    def et_path(self):
        """
        Paths to participants Eye-Tracker data.

        Returns
        -------
        et_path: str
            Path in str format to the ET folder containing the Eye-Tracker data.
        """

        if self.name == 'laptop-5i5qsv76\\joaco\n':
            et_path = 'C:/Users/joaco/OneDrive - The University of Nottingham/MEGEYEHS/DATA/ET_DATA/'
        elif self.name == 'usuario\n': # Liaa Colores
            et_path = '/mnt/6a6fd40a-e256-4844-8004-0e60d95969e8/MEGEYEHS/DATA/ET_DATA/'
        else:
            et_path = 'C:/Users/lpajg1/OneDrive - The University of Nottingham/MEGEYEHS/DATA/ET_DATA/'
        return et_path


    def beh_path(self):
        """
        Paths to participants behavioural data.

        Returns
        -------
        beh_path: str
            Path in str format to the BH folder containing the behavioural data.
        """

        if self.name == 'laptop-5i5qsv76\\joaco\n':
            beh_path = 'C:/Users/joaco/OneDrive - The University of Nottingham/MEGEYEHS/DATA/BH_DATA/'
        elif self.name == 'usuario\n': # Liaa Colores
            beh_path = '/mnt/6a6fd40a-e256-4844-8004-0e60d95969e8/MEGEYEHS/DATA/BH_DATA/'
        else:
            beh_path = 'C:/Users/lpajg1/OneDrive - The University of Nottingham/MEGEYEHS/DATA/BH_DATA/'
        return beh_path


    def mri_path(self):
        """
        Paths to participants MRI data.

        Returns
        -------
        mri_path: str
            Path in str format to the MRI folder containing the MRI data.
        """

        if self.name == 'laptop-5i5qsv76\\joaco\n':
            mri_path = 'C:/Users/joaco/OneDrive - The University of Nottingham/MEGEYEHS/DATA/MRI_DATA/'
        elif self.name == 'usuario\n': # Liaa Colores
            mri_path = '/mnt/6a6fd40a-e256-4844-8004-0e60d95969e8/MEGEYEHS/DATA/MRI_DATA/'
        else:
            mri_path = 'C:/Users/lpajg1/OneDrive - The University of Nottingham/MEGEYEHS/DATA/MRI_DATA/'
        return mri_path


    def opt_path(self):
        """
        Paths to participants digitalization data.

        Returns
        -------
        opt_path: str
            Path in str format to the OPT folder containing the digitalization data.
        """

        if self.name == 'laptop-5i5qsv76\\joaco\n':
            opt_path = 'C:/Users/joaco/OneDrive - The University of Nottingham/MEGEYEHS/DATA/OPT_DATA/'
        elif self.name == 'usuario\n': # Liaa Colores
            opt_path = '/mnt/6a6fd40a-e256-4844-8004-0e60d95969e8/MEGEYEHS/DATA/OPT_DATA/'
        else:
            opt_path = 'C:/Users/lpajg1/OneDrive - The University of Nottingham/MEGEYEHS/DATA/OPT_DATA/'
        return opt_path


    def preproc_path(self):
        """
        Paths to the preprocessed data folder.

        Returns
        -------
        preproc_path: str
            Path in str format to the folder containing the preprocessed data.
        """

        if self.name == 'laptop-5i5qsv76\\joaco\n':
            preproc_path = 'C:/Users/joaco/OneDrive - The University of Nottingham/MEGEYEHS/Save/Preprocesed_Data/'
        elif self.name == 'usuario\n': # Liaa Colores
            preproc_path = '/mnt/6a6fd40a-e256-4844-8004-0e60d95969e8/MEGEYEHS/Save/Preprocesed_Data/'
        else:
            preproc_path = 'C:/Users/lpajg1/OneDrive - The University of Nottingham/Save/Preprocesed_Data/'

        # Create directory if it doesn't exist
        os.makedirs(preproc_path, exist_ok=True)

        return preproc_path


    def results_path(self):
        """
        Paths to the results folder.

        Returns
        -------
        results_path: str
            Path in str format to the folder to store the results.
        """

        if self.name == 'laptop-5i5qsv76\\joaco\n':
            results_path = 'C:/Users/joaco/OneDrive - The University of Nottingham/MEGEYEHS/Results/'
        elif self.name == 'usuario\n': # Liaa Colores
            results_path = '/mnt/6a6fd40a-e256-4844-8004-0e60d95969e8/MEGEYEHS/Results/'
        else:
            results_path = 'C:/Users/lpajg1/OneDrive - The University of Nottingham/Results/'

        # Create directory if it doesn't exist
        os.makedirs(results_path, exist_ok=True)

        return results_path

    def plots_path(self):
        """
        Paths to the plots folder.

        Returns
        -------
        plots_path: str
            Path in str format to the folder to store the results.
        """

        if self.name == 'laptop-5i5qsv76\\joaco\n':
            plots_path = 'C:/Users/joaco/OneDrive - The University of Nottingham/MEGEYEHS/Plots/'
        elif self.name == 'usuario\n': # Liaa Colores
            plots_path = '/mnt/6a6fd40a-e256-4844-8004-0e60d95969e8/MEGEYEHS/Plots/'
        else:
            plots_path = 'C:/Users/lpajg1/OneDrive - The University of Nottingham/Plots/'

        # Create directory if it doesn't exist
        os.makedirs(plots_path, exist_ok=True)

        return plots_path


    def item_pos_path(self):
        """
        Paths to the search items positions file.

        Returns
        -------
        item_pos_path: str
            Path in str format to the items positions file.
        """

        if self.name == 'laptop-5i5qsv76\\joaco\n':
            item_pos_path = 'C:/Users/joaco/OneDrive - The University of Nottingham/MEGEYEHS/DATA/pos_items_210.mat'
        elif self.name == 'usuario\n': # Liaa Colores
            item_pos_path = '/mnt/6a6fd40a-e256-4844-8004-0e60d95969e8/MEGEYEHS/DATA/pos_items_210.mat'
        else:
            item_pos_path = 'C:/Users/lpajg1/OneDrive - The University of Nottingham/DATA/pos_items_210.mat'

        return item_pos_path