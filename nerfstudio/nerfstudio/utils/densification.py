from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import torch
import numpy as np
import open3d as o3d
from pathlib import Path

from gsplat.strategy.base import Strategy
from gsplat.strategy.ops import duplicate, remove, reset_opa, split, _update_param_with_optimizer
from typing_extensions import Literal
from copy import deepcopy


def nerf_cs_to_colmap(nerf_pcd):
    applied_transform = np.array([
        [1.0, 0.0,  0.0,  0.0],
        [0.0, 0.0,  1.0,  0.0],
        [-0.0, -1.0, -0.0, -0.0],
        [0.0, 0.0,  0.0,  1.0]
    ], dtype=np.float64)

    nerf_pcd.transform(applied_transform)
    return nerf_pcd

def k_nearest_sklearn(x: torch.Tensor, k: int):
        """
            Find k-nearest neighbors using sklearn's NearestNeighbors.
        x: The data tensor of shape [num_samples, num_features]
        k: The number of neighbors to retrieve
        """
        # Convert tensor to numpy array
        x_np = x.cpu().numpy()

        # Build the nearest neighbors model
        from sklearn.neighbors import NearestNeighbors

        nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean").fit(x_np)

        # Find the k-nearest neighbors
        distances, indices = nn_model.kneighbors(x_np)

        # Exclude the point itself from the result and return
        return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)

@dataclass
class NeRFStrategy(Strategy):
    """A default strategy that follows the original 3DGS paper:

    `3D Gaussian Splatting for Real-Time Radiance Field Rendering <https://arxiv.org/abs/2308.04079>`_

    The strategy will:

    - Periodically duplicate GSs with high image plane gradients and small scales.
    - Periodically split GSs with high image plane gradients and large scales.
    - Periodically prune GSs with low opacity.
    - Periodically reset GSs to a lower opacity.

    If `absgrad=True`, it will use the absolute gradients instead of average gradients
    for GS duplicating & splitting, following the AbsGS paper:

    `AbsGS: Recovering Fine Details for 3D Gaussian Splatting <https://arxiv.org/abs/2404.10484>`_

    Which typically leads to better results but requires to set the `grow_grad2d` to a
    higher value, e.g., 0.0008. Also, the :func:`rasterization` function should be called
    with `absgrad=True` as well so that the absolute gradients are computed.

    Args:
        prune_opa (float): GSs with opacity below this value will be pruned. Default is 0.005.
        grow_grad2d (float): GSs with image plane gradient above this value will be
          split/duplicated. Default is 0.0002.
        grow_scale3d (float): GSs with 3d scale (normalized by scene_scale) below this
          value will be duplicated. Above will be split. Default is 0.01.
        grow_scale2d (float): GSs with 2d scale (normalized by image resolution) above
          this value will be split. Default is 0.05.
        prune_scale3d (float): GSs with 3d scale (normalized by scene_scale) above this
          value will be pruned. Default is 0.1.
        prune_scale2d (float): GSs with 2d scale (normalized by image resolution) above
          this value will be pruned. Default is 0.15.
        refine_scale2d_stop_iter (int): Stop refining GSs based on 2d scale after this
          iteration. Default is 0. Set to a positive value to enable this feature.
        refine_start_iter (int): Start refining GSs after this iteration. Default is 500.
        refine_stop_iter (int): Stop refining GSs after this iteration. Default is 15_000.
        reset_every (int): Reset opacities every this steps. Default is 3000.
        refine_every (int): Refine GSs every this steps. Default is 100.
        pause_refine_after_reset (int): Pause refining GSs until this number of steps after
          reset, Default is 0 (no pause at all) and one might want to set this number to the
          number of images in training set.
        absgrad (bool): Use absolute gradients for GS splitting. Default is False.
        revised_opacity (bool): Whether to use revised opacity heuristic from
          arXiv:2404.06109 (experimental). Default is False.
        verbose (bool): Whether to print verbose information. Default is False.
        key_for_gradient (str): Which variable uses for densification strategy.
          3DGS uses "means2d" gradient and 2DGS uses a similar gradient which stores
          in variable "gradient_2dgs".

    Examples:

        >>> from gsplat import DefaultStrategy, rasterization
        >>> params: Dict[str, torch.nn.Parameter] | torch.nn.ParameterDict = ...
        >>> optimizers: Dict[str, torch.optim.Optimizer] = ...
        >>> strategy = DefaultStrategy()
        >>> strategy.check_sanity(params, optimizers)
        >>> strategy_state = strategy.initialize_state()
        >>> for step in range(1000):
        ...     render_image, render_alpha, info = rasterization(...)
        ...     strategy.step_pre_backward(params, optimizers, strategy_state, step, info)
        ...     loss = ...
        ...     loss.backward()
        ...     strategy.step_post_backward(params, optimizers, strategy_state, step, info)

    """

    prune_opa: float = 0.005
    grow_grad2d: float = 0.0002
    grow_scale3d: float = 0.01
    grow_scale2d: float = 0.05
    prune_scale3d: float = 0.1
    prune_scale2d: float = 0.15
    refine_scale2d_stop_iter: int = 0
    refine_start_iter: int = 500
    refine_stop_iter: int = 15_000
    reset_every: int = 3000
    refine_every: int = 100
    pause_refine_after_reset: int = 0
    absgrad: bool = False
    revised_opacity: bool = False
    verbose: bool = False
    key_for_gradient: Literal["means2d", "gradient_2dgs"] = "means2d"

    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        """Initialize and return the running state for this strategy.

        The returned state should be passed to the `step_pre_backward()` and
        `step_post_backward()` functions.
        """
        # Postpone the initialization of the state to the first step so that we can
        # put them on the correct device.
        # - grad2d: running accum of the norm of the image plane gradients for each GS.
        # - count: running accum of how many time each GS is visible.
        # - radii: the radii of the GSs (normalized by the image resolution).
        state = {"grad2d": None, "count": None, "scene_scale": scene_scale}
        if self.refine_scale2d_stop_iter > 0:
            state["radii"] = None
        return state

    def check_sanity(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
    ):
        """Sanity check for the parameters and optimizers.

        Check if:
            * `params` and `optimizers` have the same keys.
            * Each optimizer has exactly one param_group, corresponding to each parameter.
            * The following keys are present: {"means", "scales", "quats", "opacities"}.

        Raises:
            AssertionError: If any of the above conditions is not met.

        .. note::
            It is not required but highly recommended for the user to call this function
            after initializing the strategy to ensure the convention of the parameters
            and optimizers is as expected.
        """

        super().check_sanity(params, optimizers)
        # The following keys are required for this strategy.
        for key in ["means", "scales", "quats", "opacities"]:
            assert key in params, f"{key} is required in params but missing."

    def step_pre_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
    ):
        """Callback function to be executed before the `loss.backward()` call."""
        assert (
            self.key_for_gradient in info
        ), "The 2D means of the Gaussians is required but missing."
        info[self.key_for_gradient].retain_grad()

    def step_post_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        packed: bool = False,
    ):
        """Callback function to be executed after the `loss.backward()` call."""
        if step >= self.refine_stop_iter:
            return

        self._update_state(params, state, info, packed=packed)

        if (
            step > self.refine_start_iter
            and step % self.refine_every == 0
            and step % self.reset_every >= self.pause_refine_after_reset
        ):
            # grow GSs
            n_dupli, n_split = self._grow_gs(params, optimizers, state, step)
            if self.verbose:
                print(
                    f"Step {step}: {n_dupli} GSs duplicated, {n_split} GSs split. "
                    f"Now having {len(params['means'])} GSs."
                )

            # prune GSs
            n_prune = self._prune_gs(params, optimizers, state, step)
            if self.verbose:
                print(
                    f"Step {step}: {n_prune} GSs pruned. "
                    f"Now having {len(params['means'])} GSs."
                )

            # reset running stats
            state["grad2d"].zero_()
            state["count"].zero_()
            if self.refine_scale2d_stop_iter > 0:
                state["radii"].zero_()
            torch.cuda.empty_cache()

        if step % self.reset_every == 0:
            reset_opa(
                params=params,
                optimizers=optimizers,
                state=state,
                value=self.prune_opa * 2.0,
            )

    def _update_state(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        state: Dict[str, Any],
        info: Dict[str, Any],
        packed: bool = False,
    ):
        for key in [
            "width",
            "height",
            "n_cameras",
            "radii",
            "gaussian_ids",
            self.key_for_gradient,
        ]:
            assert key in info, f"{key} is required but missing."

        # normalize grads to [-1, 1] screen space
        if self.absgrad:
            grads = info[self.key_for_gradient].absgrad.clone()
        else:
            grads = info[self.key_for_gradient].grad.clone()
        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]

        # initialize state on the first run
        n_gaussian = len(list(params.values())[0])

        if state["grad2d"] is None:
            state["grad2d"] = torch.zeros(n_gaussian, device=grads.device)
        if state["count"] is None:
            state["count"] = torch.zeros(n_gaussian, device=grads.device)
        if self.refine_scale2d_stop_iter > 0 and state["radii"] is None:
            assert "radii" in info, "radii is required but missing."
            state["radii"] = torch.zeros(n_gaussian, device=grads.device)

        # update the running state
        if packed:
            # grads is [nnz, 2]
            gs_ids = info["gaussian_ids"]  # [nnz]
            radii = info["radii"]  # [nnz]
        else:
            # grads is [C, N, 2]
            sel = info["radii"] > 0.0  # [C, N]
            gs_ids = torch.where(sel)[1]  # [nnz]
            grads = grads[sel]  # [nnz, 2]
            radii = info["radii"][sel]  # [nnz]

        state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
        state["count"].index_add_(
            0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32)
        )
        if self.refine_scale2d_stop_iter > 0:
            # Should be ideally using scatter max
            state["radii"][gs_ids] = torch.maximum(
                state["radii"][gs_ids],
                # normalize radii to [0, 1] screen space
                radii / float(max(info["width"], info["height"])),
            )

    @torch.no_grad()
    def _grow_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
    ) -> Tuple[int, int]:
        count = state["count"]
        grads = state["grad2d"] / count.clamp_min(1)
        device = grads.device

        is_grad_high = grads > self.grow_grad2d
        is_small = (
            torch.exp(params["scales"]).max(dim=-1).values
            <= self.grow_scale3d * state["scene_scale"]
        )
        is_dupli = is_grad_high & is_small
        n_dupli = is_dupli.sum().item()

        is_large = ~is_small
        is_split = is_grad_high & is_large
        if step < self.refine_scale2d_stop_iter:
            is_split |= state["radii"] > self.grow_scale2d
        n_split = is_split.sum().item()

        # first duplicate
        if n_dupli > 0:
            n_dupli = 100000
            #duplicate(params=params, optimizers=optimizers, state=state, mask=is_dupli)
            # here we proceed with our custom implementation - densification with Nerfs
            """we propose a pipeline as follows: 
            1. Identify regions with high gradients and small scales.
            2. Load a pretrained Nerfacto model and use it to densify the identified regions.
            3. Convert the sampled points to Gaussians and add them to the scene."""

            # 1. First iteration we work with a NeRF trained on the entire scene
            print("ATTENTION Densifying with NeRF!!!")
            from nerfstudio.utils.eval_utils import eval_setup
            from nerfstudio.exporter.exporter_utils import generate_point_cloud
            from nerfstudio.scripts.exporter import ExportPointCloud, validate_pipeline
            from nerfstudio.models.splatfacto import random_quat_tensor, num_sh_bases, RGB2SH
            from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager

            nerf_config = Path("/home/team5/project/outputs/alameda/nerfacto/global_nerf/config.yml")
            exporter = ExportPointCloud(load_config=nerf_config, output_dir=None)
            _, densification_pipeline, _, _ = eval_setup(exporter.load_config)

            validate_pipeline(exporter.normal_method, exporter.normal_output_name, densification_pipeline)

            assert isinstance(densification_pipeline.datamanager, (ParallelDataManager))
            assert densification_pipeline.datamanager.train_pixel_sampler is not None
            
            # set the number of rays per batch to the number of rays per batch in the exporter
            densification_pipeline.datamanager.train_pixel_sampler.num_rays_per_batch = exporter.num_rays_per_batch
            print(f"adding {n_dupli} new points to the scene")

            pcd = generate_point_cloud(
            pipeline=densification_pipeline,
            num_points=n_dupli, #maybe change later
            remove_outliers=exporter.remove_outliers,
            reorient_normals=exporter.reorient_normals,
            estimate_normals=False,
            rgb_output_name=exporter.rgb_output_name,
            depth_output_name=exporter.depth_output_name,
            normal_output_name=exporter.normal_output_name if exporter.normal_method == "model_output" else None,
            crop_obb=None,
            std_ratio=exporter.std_ratio,
            )

            # apply the inverse dataparser transform to the point cloud
            points = np.asarray(pcd.points)
            poses = np.eye(4, dtype=np.float32)[None, ...].repeat(points.shape[0], axis=0)[:, :3, :]
            poses[:, :3, 3] = points
            poses = densification_pipeline.datamanager.train_dataparser_outputs.transform_poses_to_original_space(
                torch.from_numpy(poses)
            )
            points = poses[:, :3, 3].numpy()
            pcd.points = o3d.utility.Vector3dVector(points)

            # 2. Convert the sampled points to Gaussians and add them to the scene

            # Convert to Splatfacto Coordinates and extract coordinates and colors
            pcd = nerf_cs_to_colmap(pcd)

            # Ensure points and colors are tensors
            points_tensor = torch.Tensor(np.asarray(pcd.points))
            colors_tensor = torch.Tensor(np.asarray(pcd.colors))
            pcd = (points_tensor, colors_tensor)

            # Convert to Gaussians
            means = torch.nn.Parameter(pcd[0])
            
            distances, _ = k_nearest_sklearn(means.data, 3) # we can increase the number of neighbors here
            distances = torch.from_numpy(distances)
            # find the average of the three nearest neighbors for each point and use that as the scale
            avg_dist = distances.mean(dim=-1, keepdim=True)
            scales = torch.nn.Parameter(torch.log(avg_dist.repeat(1, 3)))
            num_points = means.shape[0]
            quats = torch.nn.Parameter(random_quat_tensor(num_points))
            dim_sh =num_sh_bases(3)

            if pcd[1].shape[0] > 0:
                shs = torch.zeros((pcd[1].shape[0], dim_sh, 3)).float().cuda()
                shs[:, 0, :3] = RGB2SH(pcd[1] / 255)
                shs[:, 1:, 3:] = 0.0
                features_dc = torch.nn.Parameter(shs[:, 0, :])
                features_rest = torch.nn.Parameter(shs[:, 1:, :])
            else:
                features_dc = torch.nn.Parameter(torch.rand(num_points, 3))
                features_rest = torch.nn.Parameter(torch.zeros((num_points, dim_sh - 1, 3)))
            
            opacities = torch.nn.Parameter(torch.logit(0.1 * torch.ones(num_points, 1)))
            print("opacities:")
            print(torch.isfinite(opacities).all())  # Should print True
            new_gauss_params = torch.nn.ParameterDict(
                {
                    "means": means,
                    "scales": scales,
                    "quats": quats,
                    "features_dc": features_dc,
                    "features_rest": features_rest,
                    "opacities": opacities,
                }
            )

            # load new gaussians to the device
            for key, value in new_gauss_params.items():
                new_gauss_params[key] = value.to(device)

            print("check device:")
            print(new_gauss_params["opacities"].device, params["opacities"].device)

            # update the parameters
            def param_fn(name: str, p: torch.Tensor) -> torch.Tensor:
                new_p = new_gauss_params[name]
                return torch.nn.Parameter(torch.cat([p, new_p]))

            def optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
                return torch.cat([v, torch.zeros((num_points, *v.shape[1:]), device=device)])

            # update the parameters and the state in the optimizers
            _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)

            # update the extra running state
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = torch.cat((v, torch.zeros_like(v[:num_points])))
            
            print("Densification with NeRF complete!")
                
        # new GSs added by duplication will not be split
        is_split = torch.cat(
            [
                is_split,
                torch.zeros(n_dupli, dtype=torch.bool, device=device),
            ]
        )

        # then split
        if n_split > 0:
            split(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=is_split,
                revised_opacity=self.revised_opacity,
            )
        return n_dupli, n_split

    @torch.no_grad()
    def _prune_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
    ) -> int:
        # Debug: Print the values of params["opacities"]
        print("params['opacities']:", params["opacities"])
        print("params['opacities'].flatten():", params["opacities"].flatten())
        print("torch.sigmoid(params['opacities'].flatten()):", torch.sigmoid(params["opacities"].flatten()))
        print("self.prune_opa:", self.prune_opa)
        is_prune = torch.sigmoid(params["opacities"].flatten()) < self.prune_opa

        if step > self.reset_every:
            is_too_big = (
                torch.exp(params["scales"]).max(dim=-1).values
                > self.prune_scale3d * state["scene_scale"]
            )
            # The official code also implements sreen-size pruning but
            # it's actually not being used due to a bug:
            # https://github.com/graphdeco-inria/gaussian-splatting/issues/123
            # We implement it here for completeness but set `refine_scale2d_stop_iter`
            # to 0 by default to disable it.
            if step < self.refine_scale2d_stop_iter:
                is_too_big |= state["radii"] > self.prune_scale2d

            is_prune = is_prune | is_too_big

        n_prune = is_prune.sum().item()
        if n_prune > 0:
            remove(params=params, optimizers=optimizers, state=state, mask=is_prune)

        return n_prune
