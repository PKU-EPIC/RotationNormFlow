import torch
from torch import nn

import pytorch3d.transforms as pttf
from flow.condition import ConditionalTransform


class UnconditionRot(nn.Module):
    def __init__(self):
        super().__init__()
        self.rot = nn.Parameter(torch.randn(
            (1, 4, 4))*1e-3+torch.eye(4).unsqueeze(0))

    def forward(self, rotation, permute=None, feature=None):
        U, S, V = torch.svd(self.rot)
        rot_mat = U.transpose(-1, -2) @ V
        rot_shape = rotation.shape
        quat = pttf.matrix_to_quaternion(rotation.reshape(-1, 3, 3))
        rotated_quat = (rot_mat @ (quat.reshape(-1, 4, 1))).reshape(-1, 4)
        result = pttf.quaternion_to_matrix(rotated_quat).reshape(rot_shape)
        return result, torch.zeros(
            (rotation.shape[0],), device=rotation.device, dtype=rotation.dtype
        )

    def inverse(self, rotation, permute=None, feature=None):
        U, S, V = torch.svd(self.rot)
        rot_mat = (U.transpose(-1, -2) @ V).transpose(-1, -2)
        rot_shape = rotation.shape
        quat = pttf.matrix_to_quaternion(rotation.reshape(-1, 3, 3))
        rotated_quat = (rot_mat @ (quat.reshape(-1, 4, 1))).reshape(-1, 4)
        result = pttf.quaternion_to_matrix(rotated_quat).reshape(rot_shape)
        return result, torch.zeros(
            (rotation.shape[0],), device=rotation.device, dtype=rotation.dtype
        )


class ConditionRot(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.net = ConditionalTransform(feature_dim, 16)

    def forward(self, rotation, permute=None, feature=None):
        mat_4d = self.net(feature).reshape(-1, 4, 4) + \
            torch.eye(4, device=rotation.device).unsqueeze(0)
        U, S, V = torch.svd(mat_4d)
        rot_mat = U.transpose(-1, -2) @ V
        rot_shape = rotation.shape
        quat = pttf.matrix_to_quaternion(rotation.reshape(-1, 3, 3))
        rotated_quat = (rot_mat @ (quat.reshape(-1, 4, 1))).reshape(-1, 4)
        result = pttf.quaternion_to_matrix(rotated_quat).reshape(rot_shape)
        return result, torch.zeros(
            (rotation.shape[0],), device=rotation.device, dtype=rotation.dtype
        )

    def inverse(self, rotation, permute=None, feature=None):
        mat_4d = self.net(feature).reshape(-1, 4, 4) + \
            torch.eye(4, device=rotation.device).unsqueeze(0)
        U, S, V = torch.svd(mat_4d)
        rot_mat = (U.transpose(-1, -2) @ V).transpose(-1, -2)
        rot_shape = rotation.shape
        quat = pttf.matrix_to_quaternion(rotation.reshape(-1, 3, 3))
        rotated_quat = (rot_mat @ (quat.reshape(-1, 4, 1))).reshape(-1, 4)
        result = pttf.quaternion_to_matrix(rotated_quat).reshape(rot_shape)
        return result, torch.zeros(
            (rotation.shape[0],), device=rotation.device, dtype=rotation.dtype
        )


def calculate_9_l(mat, rot):
    new_mat = torch.einsum('nab,nbc->nac', mat, rot)
    U, S, V = torch.svd(new_mat)
    return torch.einsum('nab,nbc->nac', U, V.transpose(-1, -2)), torch.zeros((rot.shape[0],), device=rot.device, dtype=rot.dtype)


def calculate_9_r(mat, rot):
    new_mat = torch.einsum('nab,nbc->nac', rot, mat)
    U, S, V = torch.svd(new_mat)
    return torch.einsum('nab,nbc->nac', U, V.transpose(-1, -2)), torch.zeros((rot.shape[0],), device=rot.device, dtype=rot.dtype)


def calculate_9_r_smith(mat, rot, inverse=False):
    mat_0 = mat[..., 0] / mat[..., 0].norm(dim=-1, keepdim=True)
    mat_1 = mat[..., 1] - (mat_0 * mat[..., 1]
                           ).sum(dim=-1, keepdim=True) * mat_0
    mat_1 = mat_1 / mat_1.norm(dim=-1, keepdim=True)
    mat_2 = torch.cross(mat_0, mat_1)
    new_mat = torch.cat(
        [mat_0.unsqueeze(-1), mat_1.unsqueeze(-1), mat_2.unsqueeze(-1),], dim=-1)
    if inverse:
        new_mat = new_mat.transpose(-1, -2)
    return torch.einsum('nab,nbc->nac', rot, new_mat), torch.zeros((rot.shape[0],), device=rot.device, dtype=rot.dtype)


class Uncondition9RotL(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mat = nn.Parameter(torch.eye(3)+torch.randn(3, 3)*1e-3)

    def forward(self, rotation, permute=None, feature=None):
        mat = self.mat.unsqueeze(0)
        return calculate_9_l(mat, rotation)

    def inverse(self, rotation, permute=None, feature=None):
        mat = self.mat.transpose(-1, -2).unsqueeze(0)
        return calculate_9_l(mat, rotation)


class Condition9RotL(nn.Module):
    def __init__(self, feature_dim) -> None:
        super().__init__()
        self.net = ConditionalTransform(feature_dim, 9)

    def forward(self, rotation, permute=None, feature=None):
        mat = self.net(feature).reshape(-1, 3, 3) + \
            torch.eye(3, device=rotation.device).unsqueeze(0)
        return calculate_9_l(mat, rotation)

    def inverse(self, rotation, permute=None, feature=None):
        mat = (self.net(feature).reshape(-1, 3, 3)+torch.eye(3,
               device=rotation.device).unsqueeze(0)).transpose(-1, -2)
        return calculate_9_l(mat, rotation)


class Uncondition9RotR(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mat = nn.Parameter(torch.eye(3)+torch.randn(3, 3)*1e-3)

    def forward(self, rotation, permute=None, feature=None):
        mat = self.mat.unsqueeze(0)
        return calculate_9_r(mat, rotation)

    def inverse(self, rotation, permute=None, feature=None):
        mat = self.mat.transpose(-1, -2).unsqueeze(0)
        return calculate_9_r(mat, rotation)


class Condition9RotR(nn.Module):
    def __init__(self, feature_dim) -> None:
        super().__init__()
        self.net = ConditionalTransform(feature_dim, 9)

    def forward(self, rotation, permute=None, feature=None):
        mat = self.net(feature).reshape(-1, 3, 3) + \
            torch.eye(3, device=rotation.device).unsqueeze(0)
        return calculate_9_r(mat, rotation)

    def inverse(self, rotation, permute=None, feature=None):
        mat = (self.net(feature).reshape(-1, 3, 3)+torch.eye(3,
               device=rotation.device).unsqueeze(0)).transpose(-1, -2)
        return calculate_9_r(mat, rotation)


class Uncondition9RotRSmith(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mat = nn.Parameter(torch.eye(3)+torch.randn(3, 3)*1e-3)

    def forward(self, rotation, permute=None, feature=None):
        mat = self.mat.unsqueeze(0)
        return calculate_9_r_smith(mat, rotation)

    def inverse(self, rotation, permute=None, feature=None):
        mat = self.mat.unsqueeze(0)
        return calculate_9_r_smith(mat, rotation, inverse=True)


class Condition9RotRSmith(nn.Module):
    def __init__(self, feature_dim) -> None:
        super().__init__()
        self.net = ConditionalTransform(feature_dim, 9)

    def forward(self, rotation, permute=None, feature=None):
        mat = self.net(feature).reshape(-1, 3, 3) + \
            torch.eye(3, device=rotation.device).unsqueeze(0)
        return calculate_9_r_smith(mat, rotation)

    def inverse(self, rotation, permute=None, feature=None):
        mat = self.net(feature).reshape(-1, 3, 3) + \
            torch.eye(3, device=rotation.device).unsqueeze(0)
        return calculate_9_r_smith(mat, rotation, inverse=True)
