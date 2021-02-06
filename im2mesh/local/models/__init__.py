import numpy as np
import torch
import torch.nn as nn
import trimesh
from im2mesh.local.models import decoder
from im2mesh.common import (
    get_mask, image_points_to_world, origin_to_world, normalize_tensor, calculate_berycentric_coords)


# Decoder dictionary
decoder_dict = {
    'simple': decoder.Decoder,
}


class LocalMesh(nn.Module):
    ''' DVR model class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        device (device): torch device
        depth_function_kwargs (dict): keyworded arguments for the
            depth_function
    '''

    def __init__(self, mesh: trimesh.Trimesh, decoder, encoder=None,
                 feat_dim=32, encoding_dim=128,
                 device=None):
        super().__init__()
        self.mesh = mesh
        self.verts = torch.from_numpy(mesh.vertices).requires_grad_()
        self.feats = torch.randn(
            (self.verts.shape[0], feat_dim)).requires_grad_()
        self.B = torch.randn((encoding_dim, 3))
        self.decoder = decoder.to(device)

        if encoder is not None:
            self.encoder = encoder.to(device)
        else:
            self.encoder = None

        self._device = device

    def forward(self, pixels, inputs, camera_mat,
                world_mat, scale_mat, sparse_depth=None,
                calc_deformation=False):
        ''' Performs a forward pass through the network.

        This function evaluates the depth and RGB color values for respective
        points as well as the occupancy values for the points of the helper
        losses. By wrapping everything in the forward pass, multi-GPU training
        is enabled.

        Args:
            pixels (tensor): sampled pixels
            p_occupancy (tensor): points for occupancy loss
            p_freespace (tensor): points for freespace loss
            inputs (tensor): input
            camera_mat (tensor): camera matrices
            world_mat (tensor): world matrices
            scale_mat (tensor): scale matrices
            it (int): training iteration (used for ray sampling scheduler)
            sparse_depth (dict): if not None, dictionary with sparse depth data
            calc_normals (bool): whether to calculate normals for surface
                points and a randomly-sampled neighbor
        '''
        # encode inputs
        c = self.encode_inputs(inputs)

        # transform pixels p to world
        p_world, rgb_pred, mask_pred, coords, deform, feat = self.pixels_to_world(
            pixels, camera_mat, world_mat, scale_mat, c, calc_deformation)

        if calc_deformation:
            deform_neighbor = self.get_deformation(coords, feat, c=c)
            deform = [deform, deform_neighbor]
        else:
            deform = None

        # Project pixels for sparse depth loss to world if dict is not None
        if sparse_depth is not None:
            p = sparse_depth['p']
            camera_mat = sparse_depth['camera_mat']
            world_mat = sparse_depth['world_mat']
            scale_mat = sparse_depth['scale_mat']
            p_world_sparse, _, mask_pred_sparse, _ = self.pixels_to_world(
                p, camera_mat, world_mat, scale_mat, c)
        else:
            p_world_sparse, mask_pred_sparse = None, None

        return (p_world, rgb_pred, mask_pred, deform, p_world_sparse, mask_pred_sparse)

    def get_normals(self, points, mask, c=None, h_sample=1e-3,
                    h_finite_difference=1e-3):
        ''' Returns the unit-length normals for points and one randomly
        sampled neighboring point for each point.

        Args:
            points (tensor): points tensor
            mask (tensor): mask for points
            c (tensor): latent conditioned code c
            h_sample (float): interval length for sampling the neighbors
            h_finite_difference (float): step size finite difference-based
                gradient calculations
        '''
        device = self._device

        if mask.sum() > 0:
            c = c.unsqueeze(1).repeat(1, points.shape[1], 1)[mask]
            points = points[mask]
            points_neighbor = points + (torch.rand_like(points) * h_sample -
                                        (h_sample / 2.))

            normals_p = normalize_tensor(
                self.get_central_difference(points, c=c,
                                            h=h_finite_difference))
            normals_neighbor = normalize_tensor(
                self.get_central_difference(points_neighbor, c=c,
                                            h=h_finite_difference))
        else:
            normals_p = torch.empty(0, 3).to(device)
            normals_neighbor = torch.empty(0, 3).to(device)

        return [normals_p, normals_neighbor]

    def get_deformation(self, coords, feat, c=None, h_sample=1e-3):
        coords_neighbor = coords + \
            (torch.rand_like(coords) * h_sample - (h_sample / 2.))
        coords_neighbor = torch.clamp_max(coords_neighbor, 0, 1)
        weighted_feat = coords[:, :, None] * feat
        coords_encoded = self.position_encoding(coords_neighbor, self.B)
        deform_neighbor = self.decoder(
            torch.cat([coords_encoded, weighted_feat], dim=-1), c=c, only_displacment=True
        )
        return deform_neighbor

    def get_central_difference(self, points, c=None, h=1e-3):
        ''' Calculates the central difference for points.

        It approximates the derivative at the given points as follows:
            f'(x) ≈ f(x + h/2) - f(x - h/2) for a small step size h

        Args:
            points (tensor): points
            c (tensor): latent conditioned code c
            h (float): step size for central difference method
        '''
        n_points, _ = points.shape
        device = self._device

        if c.shape[-1] != 0:
            c = c.unsqueeze(1).repeat(1, 6, 1).view(-1, c.shape[-1])

        # calculate steps x + h/2 and x - h/2 for all 3 dimensions
        step = torch.cat([
            torch.tensor([1., 0, 0]).view(1, 1, 3).repeat(n_points, 1, 1),
            torch.tensor([-1., 0, 0]).view(1, 1, 3).repeat(n_points, 1, 1),
            torch.tensor([0, 1., 0]).view(1, 1, 3).repeat(n_points, 1, 1),
            torch.tensor([0, -1., 0]).view(1, 1, 3).repeat(n_points, 1, 1),
            torch.tensor([0, 0, 1.]).view(1, 1, 3).repeat(n_points, 1, 1),
            torch.tensor([0, 0, -1.]).view(1, 1, 3).repeat(n_points, 1, 1)
        ], dim=1).to(device) * h / 2
        points_eval = (points.unsqueeze(1).repeat(1, 6, 1) + step).view(-1, 3)

        # Eval decoder at these points
        f = self.decoder(points_eval, c=c, only_occupancy=True,
                         batchwise=False).view(n_points, 6)

        # Get approximate derivate as f(x + h/2) - f(x - h/2)
        df_dx = torch.stack([
            (f[:, 0] - f[:, 1]),
            (f[:, 2] - f[:, 3]),
            (f[:, 4] - f[:, 5]),
        ], dim=-1)
        return df_dx

    # def decode(self, p, c=None, **kwargs):
    #     ''' Returns occupancy probabilities for the sampled points.

    #     Args:
    #         p (tensor): points
    #         c (tensor): latent conditioned code c
    #     '''

    #     logits = self.decoder(p, c, only_occupancy=True, **kwargs)
    #     p_r = dist.Bernoulli(logits=logits)
    #     return p_r

    def pixels_to_world(self, pixels, camera_mat, world_mat, scale_mat, c, return_deform=False):
        ''' Projects pixels to the world coordinate system.

        '''
        device = self._device
        batch_size, n_p, _ = pixels.shape

        pixels_world = image_points_to_world(pixels, camera_mat, world_mat,
                                             scale_mat)
        camera_world = origin_to_world(n_p, camera_mat, world_mat,
                                       scale_mat)
        ray_vector = (pixels_world - camera_world)

        location, index_ray, index_tri = self.mesh.ray.intersects_location(
            ray_origins=camera_world.detach().cpu().numpy().reshape(batch_size * n_p, 3),
            ray_directions=ray_vector.detach().cpu().numpy().reshape(batch_size * n_p, 3),
            multiple_hits=False
        )
        mask = np.zeros(batch_size * n_p)
        mask[index_ray] = 1

        tris = self.mesh.faces[index_tri]  # H*3
        vert = self.vert[tris]  # H*3*3
        feat = self.feat[tris]  # H*3*F

        location = torch.from_numpy(location).to(device)
        coords = calculate_berycentric_coords(location, vert)
        weighted_feat = coords[:, :, None] * feat

        coords_encoded = self.position_encoding(coords, self.B)
        out = self.decoder(
            torch.cat([coords_encoded, weighted_feat], dim=-1), c=c
        )

        p_world = camera_world.clone().view(batch_size * n_p, 3)
        c_world = torch.zeros_like(p_world)
        p_world[mask] = location + out[:, 0:3]
        c_world[mask] = torch.sigmoid(out[:, 3:6])

        p_world = p_world.view(batch_size, n_p, 3)
        c_world = c_world.view(batch_size, n_p, 3)
        mask = mask.reshape(batch_size, n_p)

        if return_deform:
            return p_world, c_world, mask, coords, out[:, 0:3], feat
        else:
            return p_world, c_world, mask

    # def decode_color(self, p_world, c=None, **kwargs):
    #     ''' Decodes the color values for world points.

    #     Args:
    #         p_world (tensor): world point tensor
    #         c (tensor): latent conditioned code c
    #     '''
    #     rgb_hat = self.decoder(p_world, c=c, only_texture=True)
    #     rgb_hat = torch.sigmoid(rgb_hat)
    #     return rgb_hat

    def encode_inputs(self, inputs):
        ''' Encodes the input.

        Args:
            input (tensor): the input
        '''

        if self.encoder is not None:
            c = self.encoder(inputs)
        else:
            c = torch.empty(inputs.size(0), 0).to(self._device)

        return c

    def to(self, device):
        ''' Puts the model to the device.

        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model

    def position_encoding(x, B):
        if B is None:
            return x
        else:
            B = B.to(x.device)
            x_proj = (2. * np.pi * x) @ B.t()
            return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
