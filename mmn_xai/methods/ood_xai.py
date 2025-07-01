import numpy as np

import torch
from torch.nn import functional as F


class OODXAI:

    def __init__(
            self,
            window_size: int,
            detector,
            occlusion_value=0.5,
            device: torch.device = None,
            batch_size: int = -1

    ):
        """

        Args:
            window_size (int): Size of the square occlusion window.
            detector (callable, optional): A function that takes a batch of images (shape (N, C, H, W)) and returns an
                OOD value per image.
            occlusion_value: For "constant" mode, either a numeric value or a string ("0" or "0.5") to compute a value
                from the image.
            device (torch.device, optional): Device to use (e.g. torch.device("cuda")). If None, CUDA is used if
                available.
        """
        self.detector = detector
        self.window_size = window_size
        self.occlusion_value = occlusion_value
        self.batch_size = batch_size

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device

    def __create_occlusions(self, image):
        """
        Create occluded versions of an image by sliding a window across it and
        replacing the window region with a specific value or computed patch value.

        The function returns a batch of occluded images along with (if requested)
        an occlusion map (accumulating detector outputs over occluded regions) and
        the detector outputs for each occluded image.

        Args:
            image (np.ndarray or torch.Tensor): Input image with shape (C, H, W).
                If a NumPy array is provided, it will be converted to a torch.Tensor.

        Returns:
            Occlusion map (accumulated detector outputs) with shape (C, H, W).
        """
        device = self.device
        detector = self.detector
        window_size = self.window_size
        occlusion_value = self.occlusion_value

        # Convert numpy array to torch.Tensor (if needed) and send to device.
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        image = image.to(device)

        # Get image dimensions (assumed to be (C, H, W))
        C, H, W = image.shape
        # Number of sliding window positions
        H_out = H - window_size + 1
        W_out = W - window_size + 1
        n_occlusions = H_out * W_out

        # Create a batch of copies of the original image.
        # We use .clone() to ensure each occluded image is independent.
        occluded_images = image.unsqueeze(0).repeat(n_occlusions, 1, 1, 1).clone()

        # Precompute the top-left coordinates for each occlusion patch.
        # These are all combinations of row (i) and column (j) positions.
        i_coords = torch.arange(H_out, device=device).repeat_interleave(W_out)
        j_coords = torch.arange(W_out, device=device).repeat(H_out)
        # Both i_coords and j_coords have shape (n_occlusions,)

        if occlusion_value == "mean":
            # For "mean", compute the mean of each patch in the original image.
            # We use F.unfold to extract all patches in one go.
            # First, add a batch dimension: shape (1, C, H, W)
            patches = F.unfold(image.unsqueeze(0), kernel_size=window_size)  # shape (1, C*window_size**2, n_occlusions)
            # Compute mean over each patch (averaged over C * window_size**2 elements)
            patch_means = patches.mean(dim=1)  # shape (1, n_occlusions)
            # Reshape to (n_occlusions, 1, 1, 1) (a scalar per patch)
            patch_vals = patch_means.view(n_occlusions, 1, 1, 1)

        else:
            # For "constant", determine the constant value (if provided as a string, compute it).
            if isinstance(occlusion_value, str):
                if occlusion_value == "0":
                    const_val = image.min()
                elif occlusion_value == "0.5":
                    const_val = (image.min() + image.max()) / 2
                else:
                    raise ValueError(f"Unsupported constant string: {occlusion_value}")
            else:
                const_val = torch.tensor(occlusion_value, device=device, dtype=image.dtype)
            # Create a patch tensor filled with the constant value.
            patch_vals = const_val.view(1, 1, 1, 1).expand(n_occlusions, 1, window_size, window_size)

        patch_vals_expanded = patch_vals.expand(n_occlusions, C, window_size, window_size)

        # --- Replace the corresponding window in each occluded image ---
        # We'll do this using scatter on a flattened view.
        # First, flatten the spatial dimensions of occluded_images:
        occluded_images_flat = occluded_images.view(n_occlusions, C, -1)  # shape: (n, C, H*W)

        # Compute flattened indices for each patch.
        # Create an offset grid for a window of shape (window_size, window_size).
        offset_i = torch.arange(window_size, device=device).view(window_size, 1)  # shape: (window_size, 1)
        offset_j = torch.arange(window_size, device=device).view(1, window_size)  # shape: (1, window_size)
        # For each occlusion, add the top-left coordinate:
        # rows: shape (n_occlusions, window_size, window_size)
        rows = i_coords.view(n_occlusions, 1, 1) + offset_i.view(1, window_size, 1)
        # cols: shape (n_occlusions, window_size, window_size)
        cols = j_coords.view(n_occlusions, 1, 1) + offset_j.view(1, 1, window_size)
        # Flatten the window indices to shape (n_occlusions, window_size*window_size)
        flat_indices = (rows * W + cols).view(n_occlusions, -1)  # Each value is in [0, H*W)

        # Flatten the patch values similarly:
        patch_vals_flat = patch_vals_expanded.view(n_occlusions, C, -1)

        # Scatter the patch values into the corresponding positions.
        occluded_images_flat.scatter_(2, flat_indices.unsqueeze(1).expand(-1, C, -1), patch_vals_flat)

        # Reshape back to (n_occlusions, C, H, W)
        occluded_images = occluded_images_flat.view(n_occlusions, C, H, W)

        # --- Optionally, run the detector on all occluded images ---
        if detector is not None:
            with torch.no_grad():
                # Assume detector accepts input of shape (N, C, H, W) on the given device.
                batch_size = occluded_images.shape[0] if self.batch_size == -1 else self.batch_size
                res = []
                for i in range(occluded_images.shape[0] // batch_size):
                    output_aux = detector(occluded_images[batch_size * i:batch_size * (i + 1)])
                    res.append(output_aux)
                detector_outputs = torch.stack(res, dim=0)

            # Squeeze extra dimensions if needed so that we have a 1D tensor per occlusion.
            values = detector_outputs.squeeze()
        else:
            values = torch.zeros(n_occlusions, device=device)

        ocl_map = torch.zeros((C, H, W), device=device, dtype=torch.float)

        # Accumulate the detector value into the patch region of the occlusion map,
        # but only for occlusions with a positive output.
        # We again use the flattened indices computed earlier.
        ocl_map_flat = ocl_map.view(C, -1)
        # Find occlusions with positive detector outputs.
        pos_idx = (values > 0).nonzero(as_tuple=False).squeeze()
        if pos_idx.numel() > 0:
            if pos_idx.numel() == 1:
                pos_idx = [pos_idx]
            # Loop over the (likely small number of) positive occlusions.
            for k in pos_idx:
                # Add the detector value (broadcasted to all channels) to the patch region.
                ocl_map_flat[:, flat_indices[k]] += values[k]
        # Reshape back to (C, H, W)
        ocl_map = ocl_map_flat.view(C, H, W)

        return ocl_map

    def attribute(self, inputs):
        return self.__create_occlusions(inputs)
