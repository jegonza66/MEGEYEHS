import os

class get:
    def __init__(self):
        self.name = os.popen('whoami').read()

    def ctf_path(self):
        if self.name == 'laptop-5i5qsv76\\joaco\n': # Asus Rog
            ctf_path = 'C:/Users/joaco/OneDrive - The University of Nottingham/MEGEYEHS/DATA/CTF_DATA/'
        elif self.name == 'usuario\n': # Liaa Colores
            ctf_path = '/mnt/6a6fd40a-e256-4844-8004-0e60d95969e8/MEGEYEHS/DATA/CTF_DATA/'
        else: # Notts
            ctf_path = 'C:/Users/lpajg1/OneDrive - The University of Nottingham/MEGEYEHS/DATA/CTF_DATA/'
        return ctf_path
    
    def et_path(self):
        if self.name == 'laptop-5i5qsv76\\joaco\n':
            et_path = 'C:/Users/joaco/OneDrive - The University of Nottingham/MEGEYEHS/DATA/ET_DATA/'
        elif self.name == 'usuario\n': # Liaa Colores
            et_path = '/mnt/6a6fd40a-e256-4844-8004-0e60d95969e8/MEGEYEHS/DATA/ET_DATA/'
        else:
            et_path = 'C:/Users/lpajg1/OneDrive - The University of Nottingham/MEGEYEHS/DATA/ET_DATA/'
        return et_path
    
    def beh_path(self):
        if self.name == 'laptop-5i5qsv76\\joaco\n':
            beh_path = 'C:/Users/joaco/OneDrive - The University of Nottingham/MEGEYEHS/DATA/BH_DATA/'
        elif self.name == 'usuario\n': # Liaa Colores
            beh_path = '/mnt/6a6fd40a-e256-4844-8004-0e60d95969e8/MEGEYEHS/DATA/BH_DATA/'
        else:
            beh_path = 'C:/Users/lpajg1/OneDrive - The University of Nottingham/MEGEYEHS/DATA/BH_DATA/'
        return beh_path

    def mri_path(self):
        if self.name == 'laptop-5i5qsv76\\joaco\n':
            mri_path = 'C:/Users/joaco/OneDrive - The University of Nottingham/MEGEYEHS/DATA/MRI_DATA/'
        elif self.name == 'usuario\n': # Liaa Colores
            mri_path = '/mnt/6a6fd40a-e256-4844-8004-0e60d95969e8/MEGEYEHS/DATA/MRI_DATA/'
        else:
            mri_path = 'C:/Users/lpajg1/OneDrive - The University of Nottingham/MEGEYEHS/DATA/MRI_DATA/'
        return mri_path

    def opt_path(self):
        if self.name == 'laptop-5i5qsv76\\joaco\n':
            opt_path = 'C:/Users/joaco/OneDrive - The University of Nottingham/MEGEYEHS/DATA/OPT_DATA/'
        elif self.name == 'usuario\n': # Liaa Colores
            opt_path = '/mnt/6a6fd40a-e256-4844-8004-0e60d95969e8/MEGEYEHS/DATA/OPT_DATA/'
        else:
            opt_path = 'C:/Users/lpajg1/OneDrive - The University of Nottingham/MEGEYEHS/DATA/OPT_DATA/'
        return opt_path