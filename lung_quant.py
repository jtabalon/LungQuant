from time import time
import argparse
from ants import from_numpy, resample_image, registration, apply_transforms
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import cm
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
from scipy.ndimage.measurements import label

from tensorflow.keras.models import load_model

from pydicom import read_file
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import imageio
import os

class LungCT():
    def __init__(self, file_path):
        self.file_path = file_path
        self.lung_volume = 0
        self.pct_emphysema = 0
        self.gas_trapped = 0
        self.perc15 = 0
        self.meanatt = 0
        self.load_ct()

    def load_ct(self):
        """ Loads in a CT dicom series and its corresponding weights

        """
        # read in dicom filenames
        dicom_files = glob(os.path.join(self.file_path, '*.dcm'))
        dicoms = [read_file(dicom_file) for dicom_file in dicom_files]

        # sort dicoms by slices
        slice_sorts = np.argsort([dicom.SliceLocation for dicom in dicoms])
        dicoms = [dicoms[slice_sort] for slice_sort in slice_sorts]

        # read in pixel array
        image = np.transpose([dicom.pixel_array * dicom.RescaleSlope + dicom.RescaleIntercept for dicom in dicoms],
                             axes=(1, 2, 0))

        # store slope/orientation to standardize image
        slope = np.float32(dicoms[1].ImagePositionPatient[2]) - \
            np.float32(dicoms[0].ImagePositionPatient[2])
        orientation = np.float32(dicoms[0].ImageOrientationPatient[4])

        # standardize image position
        if slope < 0:
            image = np.flip(image, -1)  # enforce feet first axially
        if orientation < 0:
            image = np.flip(image, 0)  # enforce supine orientation

        # extract voxel dimensions
        slice_space = np.abs(dicoms[1].SliceLocation - dicoms[0].SliceLocation)
        vox_dim = np.float32(dicoms[0].PixelSpacing[0]), np.float32(
            dicoms[0].PixelSpacing[1]), np.float32(slice_space)

        self.image = image # read in pixel array 
        self.native = image # read in native image size
        self.vox_dim = vox_dim # voxel dimensions
        self.dicoms = dicoms # read in dicoms

    def generate_masks(self, segmentation_model):
        """ Preprocess image for segmentation, predict + postprocess masks
            to respective image class.

            Args:
                segmentation_model: preloaded model used for segmentation

        """
        def preprocess_image(arr, shape=(192, 192, 192), interp=4):
            # reshape image to (192,192,192) array
            arr = from_numpy(arr)
            arr = resample_image(
                arr, shape, 1, interp_type=interp)[:, :, :]
            return arr

        def predict_segmentation(segment_model, arr):
            arr = arr / 3000.  # scale down by a factor of 3000
            # expand dimensions to (1,192,192,192,1)
            arr = np.expand_dims(np.expand_dims(arr, -1), 0)
            mask_all = segment_model.predict(
                arr)[0, :, :, :, :]  # predict masks using CNN
            return mask_all

        def postprocess_masks(mask):
            """ Postprocesses lung, lobe, and airway masks into binary masks

            Args:
                mask : (192,192,192,10) numpy array of lung,lobe,and airway masks
                        mask[192,192,192,0] : soft mask of left lung
                        mask[192,192,192,1] : soft mask of right lung
                        mask[192,192,192,2] : soft mask of fissure separation, left lung
                        mask[192,192,192,3] : soft mask of fissure separations, right lung
                        mask[192,192,192,4] : soft mask of lower lobe, left lung
                        mask[192,192,192,5] : soft mask of upper lobe, left lung
                        mask[192,192,192,6] : soft mask of lower lobe, right lung
                        mask[192,192,192,7] : soft mask of medial lobe, right lung
                        mask[192,192,192,8] : soft mask of upper lobe, right lung
                        mask[192,192,192,9] : soft mask of major airway
            Returns:
                mask_all : (192,192,192,8) numpy array of postprocessed masks
                            mask_all[192,192,192,0] : soft mask of left lung
                            mask_all[192,192,192,1] : soft mask of right lung
                            mask_all[192,192,192,2] : soft mask of lower lobe, left lung
                            mask_all[192,192,192,3] : soft mask of upper lobe, left lung
                            mask_all[192,192,192,4] : soft mask of lower lobe, right lung
                            mask_all[192,192,192,5] : soft mask of medial lobe, right lung
                            mask_all[192,192,192,6] : soft mask of upper lobe, right lung
                            mask_all[192,192,192,7] : soft mask of major airway
            """
            def categorical_masks(mask):
                """ Classifies each voxel to the most likely mask
                    (i.e. 1) finds argmax of left or right lung for lungs
                          2) finds argmax of lobes)

                    Args:
                        mask : (192,192,192,10) numpy array from lung segmentation model

                    Returns:
                        lungs : (192,192,192) numpy array of lungs with values 0, 1, 2
                                0: no class
                                1: left lung
                                2: right lung
                        lobes : (192,192,192) numpy array of lung lobes with values 0-6
                                0: no class
                                1: lower left lobe
                                2: upper left lobe
                                3: lower right lobe
                                4: medial right lobe
                                5: upper right lobe
                                6: major airway
                """
                # remove masks of fissure boundaries
                mask = mask[:, :, :, [0, 1, 4, 5, 6, 7, 8, 9]]
                mask[mask <= 0.5] = 0  # set unlikely voxels to background

                # set voxels to most likely lung class or lobe class
                s0, s1, s2, s3 = mask.shape
                lungs = np.argmax(np.concatenate(
                    [np.zeros((s0, s1, s2, 1)), mask[:, :, :, 0:2]], -1), axis=-1)
                lobes = np.argmax(np.concatenate(
                    [np.zeros((s0, s1, s2, 1)), mask[:, :, :, 2:]], -1), axis=-1)
                return lungs, lobes

            def binarize_masks(arr):
                def largest_connected_component(arr):
                    struct = np.ones((3, 3, 3))
                    labels, num_features = label(arr, struct)

                    if num_features > 0:
                        feature_size = np.zeros((num_features,))
                        for ii in range(num_features):
                            feature_size[ii] = np.sum(labels == (ii+1))

                        arr_out = np.zeros(arr.shape)
                        arr_out[labels == (np.argmax(feature_size) + 1)] = 1
                        return arr_out
                    else:
                        return arr

                nb_class = len(np.unique(arr)[1:])
                s0, s1, s2 = arr.shape
                arr_out = np.zeros([s0, s1, s2, nb_class])

                for ii in range(nb_class):
                    arr_out[:, :, :, ii] = largest_connected_component(
                        arr == ii+1)
                return arr_out

            lungs, lobes = categorical_masks(mask)
            lungs = binarize_masks(lungs)
            lobes = binarize_masks(lobes)
            mask_all = np.concatenate([lungs, lobes], -1)
            return mask_all

        if self.image.shape != (192, 192, 192):
            self.image = preprocess_image(self.image) #preprocessed image

        segmentations = predict_segmentation(segmentation_model, self.image) # initial predicted masks
        self.masks = postprocess_masks(segmentations) #categorized/processed masks

    def calculate_lung_metrics(self):
        """calculate % emphysema, % gas trapping, lung volume,
           perc15, mean attenuation

        """
        def calc_pctemph(mag, mask):
            """ percent lung <-950 """
            total_mask = np.sum(mask)
            partial_mask = np.sum(mask[(mask == 1) & (mag < -950)])
            return partial_mask / total_mask

        def calc_gastrap(mag, mask):
            """ percent lung <-950 """
            total_mask = np.sum(mask)
            partial_mask = np.sum(mask[(mask == 1) & (mag < -856)])
            return partial_mask / total_mask

        def calc_perc15(mag, mask):
            return np.percentile(mag[np.where(mask)], 15)

        def calc_lung_volume(mask, vv):
            return np.sum(mask) * vv

        def calc_meanatt(mag, mask):
            return np.mean(mag[np.where(mask)])

        def calculate_metrics(self, ct, lung_mask):
            s0, s1, s2 = self.native.shape
            m0, m1, m2 = lung_mask.shape
            vv_scaled = (s0/m0)*(s1/m1)*(s2/m2)*self.voxel_volume
            lung_volume = calc_lung_volume(lung_mask, vv_scaled)
            lung_pctemph = calc_pctemph(ct, lung_mask) * 100
            lung_pctgastrap = calc_gastrap(ct, lung_mask) * 100
            lung_perc15 = calc_perc15(ct, lung_mask)
            lung_meanatt = calc_meanatt(ct, lung_mask)

            metrics_out = dict()
            metrics_out['Volume'] = str(lung_volume / 1e6)
            metrics_out['% Emphysema'] = str(lung_pctemph)
            metrics_out['% Gas Trapping (LAA < -856 HU)'] = str(lung_pctgastrap)
            metrics_out['15th percentile'] = str(lung_perc15)
            metrics_out['Mean Attenuation'] = str(lung_meanatt)

            return metrics_out

        self.voxel_volume = np.prod(self.vox_dim) #calculated voxel volume

        copd_metrics = dict()
        # whole lung
        ct_mask = np.clip(np.sum(self.masks[:, :, :, 0:2], axis=3), 0, 1)
        metrics_whole = calculate_metrics(self, self.image, ct_mask)
        copd_metrics["Lungs"] = metrics_whole

        labels = ["Left Lung", "Right Lung",
                  "Left Lower Lobe", "Left Upper Lobe",
                  "Right Lower Lobe", "Right Middle Lobe", "Right Upper Lobe"]

        for i in range(len(labels)):
            metrics = calculate_metrics(
                self, self.image, self.masks[:, :, :, i])
            copd_metrics[labels[i]] = metrics

        self.copd_metrics = copd_metrics #dictionary containing calculated metrics

    def export_lung_metrics(self, file_prefix, output_path):
        """ Export generated metrics to .csv

            Args:
                file_prefix: name of .csv file outputs _metrics.csv

        """
        lung_metrics = pd.DataFrame.from_dict(
            self.copd_metrics, orient='index')
        print(lung_metrics)
        if output_path:
            print(f"writing {file_prefix + r'_metrics.csv'} to {output_path}")
            lung_metrics.to_csv(os.path.join(
                output_path, file_prefix + r'_metrics.csv'))
            print(f"{file_prefix + r'_metrics.csv'} exported to {output_path}")
        else:
            if not os.path.exists('metrics'):
                os.mkdir('metrics')
            print(f"writing {file_prefix + r'_metrics.csv'} to /metrics")
            lung_metrics.to_csv(os.path.join(
                'metrics', file_prefix + r'_metrics.csv'))
            print(f"{file_prefix + r'_metrics.csv'} exported to {os.getcwd() + '/metrics'}")


class Registration():
    def __init__(self, registration_model, transformer, inspiratory, expiratory):
        self.registration_model = registration_model
        self.transformer = transformer
        self.affine_registration = 0
        self.insp_mask = inspiratory.masks
        self.insp = inspiratory.image
        self.insp_dicoms = inspiratory.dicoms
        self.insp_native = inspiratory.native
        self.exp = expiratory.image
        self.exp_mask = expiratory.masks
        self.exp_dicoms = expiratory.dicoms
        self.exp_affine = 0
        self.exp_mask_affine = 0
        self.disp = 0
        self.exp_deform = 0
        self.exp_mask_deform = 0
        self.adm = 0
        self.adm_metrics = dict()

    def create_adm(self):
        def preprocess_image(self, arr, shape=(192, 192, 192), interp=4):
            arr = from_numpy(arr)
            arr = resample_image(
                arr, shape, 1, interp_type=interp)[:, :, :]
            return arr

        insp_shape = self.insp_native.shape

        insp_whole = np.clip(
            self.insp_mask[:, :, :, 0] + self.insp_mask[:, :, :, 1], 0, 1)
        insp_whole[self.insp > 0] = 0
        insp_whole[self.exp_deform > 0] = 0

        insp_slc_mask_smooth = gaussian_filter(self.insp, sigma=1)*insp_whole
        diff_slc_mask_smooth = np.abs(gaussian_filter(
            (self.exp_deform*3000 - self.insp), sigma=1))*insp_whole

        insp_whole = preprocess_image(
            self, insp_whole, shape=insp_shape, interp=0)
        insp_whole = np.clip(np.round(insp_whole), 0, 1)

        insp_slc_mask_smooth = preprocess_image(self, insp_slc_mask_smooth,
                                                shape=insp_shape,
                                                interp=0)
        diff_slc_mask_smooth = preprocess_image(self, diff_slc_mask_smooth,
                                                shape=insp_shape,
                                                interp=0)

        if not os.path.exists('adm'):
            os.mkdir('adm')

        for slc_ind in tqdm(range(insp_shape[0])):

            # extract axial slice
            insp_slc = np.rot90(self.insp_native[slc_ind, :, :])
            insp_mask = np.rot90(insp_whole[slc_ind, :, :])

            # axial slices for color capture
            insp_slc_mask = np.rot90(insp_slc_mask_smooth[slc_ind, :, :])
            diff_slc_mask = np.rot90(diff_slc_mask_smooth[slc_ind, :, :])

            # inspiratory color map
            insp_map = cm.get_cmap('Reds', 512)(np.linspace(1, 0, 512))
            insp_map[:, 3] = np.linspace(1, 0, 512)
            insp_map[:, 0:3] = insp_map[256, 0:3]
            insp_map = ListedColormap(insp_map)

            # expiratory color map
            diff_map = cm.get_cmap('Blues', 512)(np.linspace(1, 0, 512))
            diff_map[:, 3] = np.linspace(1, 0, 512)
            diff_map[:, 0:3] = diff_map[128, 0:3]
            diff_map = ListedColormap(diff_map)

            # create secondary capture plot
            f, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.imshow(insp_slc, cmap='gray', vmin=-400-1500, vmax=-400+1500)
            ax.imshow(np.ma.masked_where(insp_mask == 0, diff_slc_mask), cmap=diff_map,
                      vmin=0, vmax=100, alpha=0.5)
            ax.imshow(np.ma.masked_where(insp_mask == 0, insp_slc_mask), cmap=insp_map,
                      vmin=-1000, vmax=-925, alpha=0.5)
            plt.axis('off')
            f.savefig(os.path.join("adm", format(slc_ind, "03") +
                                   '.png'), bbox_inches='tight', format='png')
            plt.close()

        # render and save gif
        image_files = os.listdir("adm")
        images = []

        for image in sorted(image_files):
            images.append(imageio.imread(os.path.join("adm", image))) #list of images

        imageio.mimsave('adm.gif', images, duration=0.1)

    def calculate_adm_metrics(self):
        """ generate attenuation difference metrics """
        self.adm = self.exp_deform * 3000 - self.insp #calculated attenuation difference

        def calculate_adm(self, adm, mask):
            total_mask = np.sum(mask)
            partial_mask = np.sum(mask[(mask == 1) & (adm < 100)]) #iterated mask
            return str(partial_mask / total_mask * 100)

        def calc_adm_metrics(self, adm, mask):
            adm_metrics = dict()
            labels = ["Left Lung", "Right Lung",
                      "Left Lower Lobe", "Left Upper Lobe",
                      "Right Lower Lobe", "Right Middle Lobe", "Right Upper Lobe"]
            # whole lung

            mask_in = np.clip(np.sum(mask[:, :, :, 0:2], axis=3), 0, 1)
            metrics_whole = calculate_adm(self, self.adm, mask_in)
            adm_metrics["Lungs"] = metrics_whole

            # each lung and lung lobe
            for i in range(len(labels)):
                metrics = calculate_adm(self, self.adm, mask[:, :, :, i])
                adm_metrics[labels[i]] = metrics

            self.adm_metrics = adm_metrics #dictionary containing adm metrics

            return adm_metrics
        calc_adm_metrics(self, self.adm, self.insp_mask)

    def export_metrics(self, file_prefix, output_path):
        """ Export generated metrics to .csv
            Args:
                file_prefix: name of .csv file outputs _metrics.csv
        """
        registration_metrics = pd.DataFrame.from_dict(
            self.adm_metrics, orient='index', columns=['Gas Trapping'])
        print(registration_metrics)
        if output_path:
            print(f"writing {file_prefix + r'_metrics.csv'} to {output_path}")
            registration_metrics.to_csv(os.path.join(
                output_path, file_prefix + r'_metrics.csv'))
            print(f"{file_prefix + r'_metrics.csv'} exported to {output_path}")
        else:
            if not os.path.exists('metrics'):
                os.mkdir('metrics')
            print(f"writing {file_prefix + r'_metrics.csv'} to /metrics")
            registration_metrics.to_csv(os.path.join(
                'metrics', file_prefix + r'_metrics.csv'))
            print(f"{file_prefix + r'_metrics.csv'} exported to {os.getcwd() + '/metrics'}")

    def register(self):
        """ Register expiratory to inspiratory images using affine,
                then deformable registration """
        def lungQuant_registration(self, lungreg_model, lungreg_transform, insp, exp, insp_mask, exp_mask):
            """ Register expiratory to inspiratory images """

            def affine_registration(self, fixed, moving):
                """ Affine registration of a moving image to a fixed image"""
                affine_reg = registration(fixed=from_numpy(fixed),
                                               moving=from_numpy(moving),
                                               type_of_transform='AffineFast')
                return affine_reg

            def affine_transform(self, affine_reg, insp, exp, insp_mask, exp_mask):
                """ Affine registration of expiratory image and lung masks to
                    inspiratory image
                """
                # affine register image
                exp_affine = apply_transforms(fixed=from_numpy(insp),
                                                   moving=from_numpy(exp),
                                                   transformlist=affine_reg['fwdtransforms'])[:, :, :]
                # affine register lobar masks
                exp_mask_affine = np.zeros(exp_mask.shape)
                for i in range(exp_mask.shape[3]):
                    exp_mask_affine[:, :, :, i] = apply_transforms(fixed=from_numpy(exp_mask[:, :, :, i]),
                                                                        moving=from_numpy(
                                                                            exp_mask[:, :, :, i]),
                                                                        transformlist=affine_reg['fwdtransforms'])[:, :, :]
                    exp_mask_affine[exp_mask_affine > 0.5] = 1 #binarize
                    exp_mask_affine[exp_mask_affine <= 0.5] = 0 #binarize
                return exp_affine, exp_mask_affine

            def vm_deform(self, lungreg_model, lungreg_transform, insp, exp_affine, insp_mask, exp_mask_affine):
                # prepare images for registration
                insp_input = np.expand_dims(
                    np.expand_dims(insp / 3000., 0), -1)
                exp_input = np.expand_dims(
                    np.expand_dims(exp_affine / 3000., 0), -1)

                # extract deformed expiratory and displacement field
                exp_deform, disp = lungreg_model.predict(
                    [exp_input, insp_input])

                exp_mask_deform = np.zeros(exp_mask_affine.shape)
                for i in range(exp_mask_affine.shape[3]):
                    exp_input = np.expand_dims(np.expand_dims(
                        exp_mask_affine[:, :, :, i], 0), -1)
                    exp_mask_deform[:, :, :, i] = lungreg_transform.predict([exp_input, disp])[
                        0, :, :, :, 0]
                exp_mask_deform[exp_mask_deform > 0.5] = 1
                exp_mask_deform[exp_mask_deform <= 0.5] = 0

                return exp_deform[0, :, :, :, 0], exp_mask_deform, disp

            # affine registration
            self.affine_registration = affine_registration(self, np.clip(np.sum(
                insp_mask[:, :, :, 0:2], axis=3), 0, 1), np.clip(np.sum(exp_mask[:, :, :, 0:2], axis=3), 0, 1))

            # transform image and masks
            self.exp_affine, self.exp_mask_affine = affine_transform(
                self, self.affine_registration, self.insp, self.exp, self.insp_mask, self.exp_mask)

            # deformable registration
            self.exp_deform, self.exp_mask_deform, self.disp = vm_deform(self, self.registration_model,
                                                                         self.transformer,
                                                                         self.insp, self.exp_affine,
                                                                         self.insp_mask, self.exp_mask_affine)

        lungQuant_registration(self, self.registration_model, self.transformer,
                            self.insp, self.exp, self.insp_mask, self.exp_mask)


def main(args):

    # load models
    segmentation_model =load_model('models/segmentation_model')
    registration_model = load_model('models/registration_model')
    transformer = load_model('models/transformer_model')

    # load and preprocess images
    start = time()
    if args.insp_image_path:
        inspiratory = LungCT(args.insp_image_path)
    if args.exp_image_path:
        expiratory = LungCT(args.exp_image_path)
    inspiratory = LungCT("data/insp")
    expiratory = LungCT("data/exp")
    end = time()
    print(f"load image: {end - start} seconds")

    # create masks
    start = time()
    inspiratory.generate_masks(segmentation_model)
    expiratory.generate_masks(segmentation_model)
    end = time()
    print(f"resizing images + segmentation: {end - start}")

    # register images
    start = time()
    registration = Registration(
        registration_model, transformer, inspiratory, expiratory)
    registration.register()
    end = time()
    print(f"register images + transforming segmentations: {end - start}")

    # calculate metrics
    start = time()
    inspiratory.calculate_lung_metrics()
    expiratory.calculate_lung_metrics()
    end = time()
    print(f"calculate metrics: {end - start}")

    # export metrics
    inspiratory.export_lung_metrics('inspiratory', args.metrics_output_path)
    expiratory.export_lung_metrics('expiratory', args.metrics_output_path)

    # create adm
    registration.calculate_adm_metrics()
    registration.export_metrics('registration', args.metrics_output_path)
    registration.create_adm()

if '__name__==__main__':

    # Instantiate an arg parser
    parser = argparse.ArgumentParser()

    # Set arguments and their default values

    parser.add_argument(
        "-i",
        "--insp_image_path",
        type=str,
        default=None,
        help="Path to inspiratory dicom series (.dcm)"
    )

    parser.add_argument(
        "-e",
        "--exp_image_path",
        type=str,
        default=None,
        help="Path to expiratory dicom series (.dcm)"
    )

    parser.add_argument(
        "-m",
        "--metrics_output_path",
        type=str,
        default=None,
        help="Path to save metrics."
    )


    parser.add_argument(
        "-a",
        "--adm_map",
        action="store_true", #ask about this
        default=True,
        help="Generate Attenuation Difference Map."
    )

    # Save command line arguments
    args = parser.parse_args()

    main(args)
