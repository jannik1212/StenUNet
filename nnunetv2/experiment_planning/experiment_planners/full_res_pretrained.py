from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner
from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape
import numpy as np
import shutil
from os.path import join
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed


class StenUNetFullresOnlyPlanner(ExperimentPlanner):
    def plan_experiment(self):
        """
        Only generate 3d_fullres configuration and skip 2D and lowres/cascade.
        """
        transpose_forward, transpose_backward = self.determine_transpose()
        fullres_spacing = self.determine_fullres_target_spacing()
        fullres_spacing_transposed = fullres_spacing[transpose_forward]

        new_shapes = [
            compute_new_shape(i, j, fullres_spacing)
            for i, j in zip(self.dataset_fingerprint['spacings'], self.dataset_fingerprint['shapes_after_crop'])
        ]

        new_median_shape = np.median(new_shapes, 0)
        new_median_shape_transposed = new_median_shape[transpose_forward]

        approximate_n_voxels_dataset = float(
            np.prod(new_median_shape_transposed, dtype=np.float64) * self.dataset_json['numTraining']
        )

        plan_3d_fullres = self.get_plans_for_configuration(
            fullres_spacing_transposed,
            new_median_shape_transposed,
            self.generate_data_identifier("3d_fullres"),
            approximate_n_voxels_dataset,
        )

        plan_3d_fullres['batch_dice'] = False

        median_spacing = np.median(self.dataset_fingerprint['spacings'], 0)[transpose_forward]
        median_shape = np.median(self.dataset_fingerprint['shapes_after_crop'], 0)[transpose_forward]

        self.plans = {
            'dataset_name': self.dataset_name,
            'plans_name': self.plans_identifier,
            'original_median_spacing_after_transp': [float(i) for i in median_spacing],
            'original_median_shape_after_transp': [int(round(i)) for i in median_shape],
            'image_reader_writer': self.determine_reader_writer().__name__,
            'transpose_forward': [int(i) for i in transpose_forward],
            'transpose_backward': [int(i) for i in transpose_backward],
            'configurations': {'3d_fullres': plan_3d_fullres},
            'experiment_planner_used': self.__class__.__name__,
            'label_manager': 'LabelManager',
            'foreground_intensity_properties_per_channel': self.dataset_fingerprint[
                'foreground_intensity_properties_per_channel']
        }
        
        shutil.copy(join(self.raw_dataset_folder, 'dataset.json'),
            join(nnUNet_preprocessed, self.dataset_name, 'dataset.json'))

        self.save_plans(self.plans)
        return self.plans

    def get_plans_for_configuration(self, spacing, median_shape, data_identifier, approximate_n_voxels_dataset):
        plan = super().get_plans_for_configuration(spacing, median_shape, data_identifier, approximate_n_voxels_dataset)

        # ✅ Manually truncate to match the pretrained encoder (4 stages)
        plan['n_conv_per_stage_encoder'] = (2, 2, 2, 2)
        plan['n_conv_per_stage_decoder'] = (2, 2, 2)
        plan['num_pool_per_axis'] = [3, 3, 3]
        plan['pool_op_kernel_sizes'] = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 1, 1]]
        plan['conv_kernel_sizes'] = [[3, 3, 3]] * 4

        return plan