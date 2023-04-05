import matplotlib.pyplot as plt
import setup
import load
from paths import paths
import pathlib
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D
import os
from scipy import ndimage

# Load experiment info
exp_info = setup.exp_info()

mri_path = paths().mri_path()
subject = load.preproc_subject(exp_info=exp_info, subject_code=0)


subj_path = pathlib.Path(os.path.join(mri_path, subject.subject_id))
mri_file_path = list(subj_path.glob('*.nii'))[1]
img = nib.load(mri_file_path)
hdr = img.header
hdr.get_xyzt_units()
img_data = img.get_fdata()

## 3 views plot
fig, ax = plt.subplots(ncols=3, figsize=(15, 5))

slice_1 = 80
ax[0].imshow(img_data[slice_1, :, :].T, origin='lower', cmap='gray')
ax[0].set_xlabel('Second dim voxel coords.', fontsize=12)
ax[0].set_ylabel('Third dim voxel coords', fontsize=12)
ax[0].set_title('First dimension, slice nr. {}'.format(slice_1), fontsize=15)

slice_2 = 100
ax[1].imshow(img_data[:, slice_2, :].T, origin='lower', cmap='gray')
ax[1].set_xlabel('First dim voxel coords.', fontsize=12)
ax[1].set_ylabel('Third dim voxel coords', fontsize=12)
ax[1].set_title('Second dimension, slice nr. {}'.format(slice_2), fontsize=15)

slice_3 = 160
ax[2].imshow(img_data[:, :, slice_3].T, origin='lower', cmap='gray')
ax[2].set_xlabel('First dim voxel coords.', fontsize=12)
ax[2].set_ylabel('Second dim voxel coords', fontsize=12)
ax[2].set_title('Third dimension, slice nr. {}'.format(slice_3), fontsize=15)

fig.tight_layout()


## Lateral dinamyc plots
fig, ax = plt.subplots(1, 1)
im = plt.imshow(img_data[0, :, :], cmap="gray")
for i in range(img_data.shape[0]):
    rotated_img = ndimage.rotate(img_data[i,:,:], 90)
    im.set_data(rotated_img)
    plt.pause(0.1)
plt.close()


## GIF
filenames = []
plt.ioff()
for i in range(img_data.shape[0]):
    plt.imshow(ndimage.rotate(img_data[i,:,:], 90), cmap="gray")
    # create file name and append it to a list
    filename = f'{i}.png'
    filenames.append(filename)
    # save frame
    plt.savefig(filename)
    plt.close()

# build gif
import imageio.v2 as imageio
with imageio.get_writer('mygif.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# Remove files
for filename in set(filenames):
    os.remove(filename)

## Visualizations
import mne
subjects_dir = os.path.join(mri_path, 'FreeSurfer_out')
os.environ["SUBJECTS_DIR"] = subjects_dir

## MNE - Freesurfer
# https://mne.tools/stable/auto_tutorials/forward/50_background_freesurfer_mne.html#sphx-glr-auto-tutorials-forward-50-background-freesurfer-mne-py

fname = os.path.join(subjects_dir, subject.subject_id, 'surf', 'rh.white')
rr_mm, tris = mne.read_surface(fname)
print(f'rr_mm.shape == {rr_mm.shape}')
print(f'tris.shape == {tris.shape}')
print(f'rr_mm.max() = {rr_mm.max()}')

renderer = mne.viz.backends.renderer.create_3d_figure(size=(600, 600), bgcolor='w', scene=False)
gray = (0.5, 0.5, 0.5)
renderer.mesh(*rr_mm.T, triangles=tris, color=gray)
view_kwargs = dict(elevation=90, azimuth=0)
mne.viz.set_3d_view(figure=renderer.figure, distance=350, focalpoint=(0., 0., 40.), **view_kwargs)
renderer.show()

## Full brain color plot
# https://mne.tools/stable/auto_tutorials/forward/10_background_freesurfer.html#sphx-glr-auto-tutorials-forward-10-background-freesurfer-py
Brain = mne.viz.get_brain_class()
brain = Brain(subject.subject_id, hemi='rh', surf='sphere', size=(800, 600))
# brain.add_annotation('aparc.a2009s', borders=False)

##
# https://mne.tools/stable/auto_examples/visualization/parcellation.html#sphx-glr-auto-examples-visualization-parcellation-py

Brain = mne.viz.get_brain_class()
mne.datasets.fetch_aparc_sub_parcellation(subjects_dir=subjects_dir, verbose=True)

# Candidate parc: aparc, aparc.a2009s, aparc.DKTatlas40, BA, BA.thresh
labels = mne.read_labels_from_annot(subject=subject.subject_id, parc='aparc', subjects_dir=subjects_dir)

brain = Brain(subject.subject_id, 'both', 'inflated', subjects_dir=subjects_dir, cortex='low_contrast', background='white',
              size=(800, 600))
# brain.add_annotation('aparc.a2009s')
# aud_label = [label for label in labels if label.name == 'bankssts-lh'][0]
# brain.add_label(aud_label, borders=False)