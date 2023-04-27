import torch
from torch import nn
import numpy as np

import pytorch3d.transforms as pttf
from flow.condition import ConditionalTransform
from scipy import linalg as la


def my_det_3_3(A):
    det00 = A[..., 1, 1]*A[..., 2, 2]-A[..., 1, 2]*A[..., 2, 1]
    det01 = A[..., 1, 2]*A[..., 2, 0]-A[..., 1, 0]*A[..., 2, 2]
    det02 = A[..., 1, 0]*A[..., 2, 1]-A[..., 1, 1]*A[..., 2, 0]
    return det00*A[..., 0, 0]+det01*A[..., 0, 1]+det02*A[..., 0, 2]


def my_det_4_4(A):
    det00 = A[..., 0, 0]*my_det_3_3(A[..., 1:, [1, 2, 3]])
    det01 = A[..., 0, 1]*my_det_3_3(A[..., 1:, [0, 2, 3]])
    det02 = A[..., 0, 2]*my_det_3_3(A[..., 1:, [0, 1, 3]])
    det03 = A[..., 0, 3]*my_det_3_3(A[..., 1:, [0, 1, 2]])
    return det00-det01+det02-det03


def batch_svd_proper(A):
    U, S, V = torch.svd(A)
    U2 = U * U.det().unsqueeze(-1).unsqueeze(-1)
    V2 = V * V.det().unsqueeze(-1).unsqueeze(-1)
    S2 = S * (U.det()*V.det()).unsqueeze(-1)
    return U2, S2, V2


def calculate_16(mat, rotation):
    quat = pttf.matrix_to_quaternion(rotation)
    quat = mat @ quat.reshape(-1, 4, 1)
    length = quat.norm(dim=-2, keepdim=True)
    t_rotation = pttf.quaternion_to_matrix((quat/length).reshape(-1, 4))
    return t_rotation, my_det_4_4(mat).abs().log()-4*length.reshape(-1).log()


class Condition16Trans(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.net = ConditionalTransform(feature_dim, 16)

    def forward(self, rotation, permute=None, feature=None):
        mat = self.net(feature).reshape(-1, 4, 4) + \
            torch.eye(4, device=rotation.device).unsqueeze(0)
        return calculate_16(mat, rotation)

    def inverse(self, rotation, permute=None, feature=None):
        mat = self.net(feature).reshape(-1, 4, 4) + \
            torch.eye(4, device=rotation.device).unsqueeze(0)
        mat = torch.linalg.inv(mat)
        return calculate_16(mat, rotation)


class UnconditionLU(nn.Module):
    # from https://github.com/rosinality/glow-pytorch/blob/master/model.py
    def __init__(self, in_channel) -> None:
        super().__init__()
        weight = 1e-3*np.random.randn(in_channel,
                                      in_channel) + np.eye(in_channel)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l)
        w_s = torch.from_numpy(np.copy(w_s))
        w_u = torch.from_numpy(w_u)

        self.register_buffer("w_p", w_p)
        self.register_buffer("u_mask", torch.from_numpy(u_mask))
        self.register_buffer("l_mask", torch.from_numpy(l_mask))
        self.register_buffer("s_sign", torch.sign(w_s))
        self.register_buffer("l_eye", torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(w_s.abs().log())
        self.w_u = nn.Parameter(w_u)

    def forward(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )
        return weight.unsqueeze(0)


class ConditionLU(nn.Module):
    def __init__(self, in_channel, feature_dim) -> None:
        super().__init__()
        self.in_channel = in_channel
        weight = np.random.randn(in_channel, in_channel)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l)
        w_s = torch.from_numpy(np.copy(w_s))
        w_u = torch.from_numpy(w_u)

        self.register_buffer("w_p", w_p)
        self.register_buffer("u_mask", torch.from_numpy(u_mask))
        self.register_buffer("l_mask", torch.from_numpy(l_mask))
        self.register_buffer("s_sign", torch.sign(w_s))
        self.register_buffer("l_eye", torch.eye(l_mask.shape[0]))
        self.w_l_net = ConditionalTransform(feature_dim, in_channel*in_channel)
        self.w_u_net = ConditionalTransform(feature_dim, in_channel*in_channel)
        self.w_s_net = ConditionalTransform(feature_dim, in_channel)

    def forward(self, feature):
        weight = torch.einsum(
            'ab,nbc,ncd->nad',
            self.w_p,
            (self.w_l_net(feature).reshape(-1, self.in_channel,
             self.in_channel) * self.l_mask + self.l_eye),
            ((self.w_u_net(feature).reshape(-1, self.in_channel, self.in_channel) * self.u_mask) +
             torch.diag(self.s_sign * torch.exp(self.w_s_net(feature))))
        )
        return weight


class Condition16TransLU(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.net = ConditionLU(4, feature_dim)

    def forward(self, rotation, permute=None, feature=None):
        mat = self.net(feature)
        return calculate_16(mat, rotation)

    def inverse(self, rotation, permute=None, feature=None):
        mat = self.net(feature)
        mat = torch.linalg.inv(mat)
        return calculate_16(mat, rotation)


class Uncondition16TransLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.mat = UnconditionLU(4)

    def forward(self, rotation, permute=None, feature=None):
        return calculate_16(self.mat(), rotation)

    def inverse(self, rotation, permute=None, feature=None):
        mat = self.mat()
        mat = torch.linalg.inv(mat)
        return calculate_16(mat, rotation)


class Uncondition16Trans(nn.Module):
    def __init__(self):  # , printW=0):
        super().__init__()
        self.mat = nn.Parameter(
            torch.eye(4).unsqueeze(0)+torch.randn(1, 4, 4)*1e-3)

    def forward(self, rotation, permute=None, feature=None):
        mat = self.mat
        return calculate_16(mat, rotation)

    def inverse(self, rotation, permute=None, feature=None):
        mat = self.mat
        mat = torch.linalg.inv(mat)
        return calculate_16(mat, rotation)


def accp_normalize(raw, raw_accp):
    raw_square_sum = raw.square().sum(dim=-1, keepdim=True)
    raw_square_sum_accp = (2 * raw_accp * raw).sum(dim=-1,
                                                   keepdim=True)  # (a+kx)^2 -> a^2 + 2kax
    raw_norm = raw_square_sum.sqrt()
    # sqrt(a+kx) -> sqrt(a) + 0.5kx/sqrt(a)
    raw_norm_accp = raw_square_sum_accp/2/raw_norm
    raw_norm_inv = 1/raw_norm
    raw_norm_inv_accp = -raw_norm_accp * \
        raw_norm_inv.square()  # 1/(a+kx) -> 1/a - kx/a^2
    return raw_norm_inv * raw, raw_norm_inv_accp * raw + raw_norm_inv * raw_accp


def accp_cross(a1, a1_accp, a2, a2_accp):
    permute_1 = torch.tensor([1, 2, 0], device=a1.device)
    permute_2 = torch.tensor([2, 0, 1], device=a1.device)
    a3 = torch.cross(a1, a2, dim=-1)
    a3_accp = a1[..., permute_1] * a2_accp[..., permute_2] + a1_accp[..., permute_1] * a2[..., permute_2] - \
        a1[..., permute_2] * a2_accp[..., permute_1] - \
        a1_accp[..., permute_2] * a2[..., permute_1]
    return a3, a3_accp


def calculate_9(mat, rotation):
    accp = torch.tensor([[[0, 1, 0], [-1, 0, 0], [0, 0, 0]], [[0, 0, 1], [0, 0, 0], [-1, 0, 0]],
                        [[0, 0, 0], [0, 0, 1], [0, -1, 0]]], dtype=mat.dtype, device=mat.device)

    rotation_mat = torch.einsum(
        'nab,nbc->nac', mat.reshape(-1, 3, 3), rotation)
    rotation_mat_accp = torch.einsum('nab,kbc->knac', rotation_mat, accp)
    rotation_mat_0 = rotation_mat[..., 0]
    rotation_mat_0_accp = rotation_mat_accp[..., 0]
    t_rotation_mat_0, t_rotation_mat_0_accp = accp_normalize(
        rotation_mat_0, rotation_mat_0_accp)
    rotation_mat_1 = rotation_mat[..., 1]
    rotation_mat_1_accp = rotation_mat_accp[..., 1]
    dot = (t_rotation_mat_0 * rotation_mat_1).sum(dim=-1, keepdim=True)
    dot_accp = (t_rotation_mat_0_accp*rotation_mat_1 +
                t_rotation_mat_0*rotation_mat_1_accp).sum(dim=-1, keepdim=True)
    raw_t_mat_1 = rotation_mat_1 - dot*t_rotation_mat_0
    raw_t_mat_1_accp = rotation_mat_1_accp - \
        (dot_accp * t_rotation_mat_0 + dot * t_rotation_mat_0_accp)
    t_rotation_mat_1, t_rotation_mat_1_accp = accp_normalize(
        raw_t_mat_1, raw_t_mat_1_accp)
    t_rotation_mat_2, t_rotation_mat_2_accp = accp_cross(
        t_rotation_mat_0, t_rotation_mat_0_accp, t_rotation_mat_1, t_rotation_mat_1_accp)
    t_rotation = torch.cat([t_rotation_mat_0.unsqueeze(-1),
                           t_rotation_mat_1.unsqueeze(-1), t_rotation_mat_2.unsqueeze(-1)], dim=-1)
    t_rotation_accp = torch.cat([t_rotation_mat_0_accp.unsqueeze(
        -1), t_rotation_mat_1_accp.unsqueeze(-1), t_rotation_mat_2_accp.unsqueeze(-1)], dim=-1)
    delta = torch.einsum('knab,nbc->knac', t_rotation_accp,
                         t_rotation.transpose(-1, -2))
    vector = delta[..., [0, 0, 1], [1, 2, 2]]

    return t_rotation, my_det_3_3(vector.transpose(0, 1)).abs().log()


class Condition9Trans(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.net = ConditionalTransform(feature_dim, 9)

    def forward(self, rotation, permute=None, feature=None):
        mat = self.net(feature).reshape(-1, 3, 3) + \
            torch.eye(3, device=rotation.device).unsqueeze(0)
        return calculate_9(mat, rotation)

    def inverse(self, rotation, permute=None, feature=None):
        mat = torch.linalg.inv(self.net(
            feature).reshape(-1, 3, 3)+torch.eye(3, device=rotation.device).unsqueeze(0))
        return calculate_9(mat, rotation)


class Uncondition9Trans(nn.Module):
    def __init__(self):
        super().__init__()
        self.mat = nn.Parameter(torch.eye(3)+torch.randn(3, 3)*1e-3)

    def forward(self, rotation, permute=None, feature=None):
        mat = self.mat
        return calculate_9(mat, rotation)

    def inverse(self, rotation, permute=None, feature=None):
        mat = torch.linalg.inv(self.mat)
        return calculate_9(mat, rotation)


class Condition9TransLU(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.net = ConditionLU(3, feature_dim)

    def forward(self, rotation, permute=None, feature=None):
        mat = self.net(feature).reshape(-1, 3, 3) + \
            torch.eye(3, device=rotation.device).unsqueeze(0)
        return calculate_9(mat, rotation)

    def inverse(self, rotation, permute=None, feature=None):
        mat = torch.linalg.inv(self.net(
            feature).reshape(-1, 3, 3)+torch.eye(3, device=rotation.device).unsqueeze(0))
        return calculate_9(mat, rotation)


class Uncondition9TransLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.mat = UnconditionLU(3)

    def forward(self, rotation, permute=None, feature=None):
        mat = self.mat()
        return calculate_9(mat, rotation)

    def inverse(self, rotation, permute=None, feature=None):
        mat = torch.linalg.inv(self.mat())
        return calculate_9(mat, rotation)


def calculate_36(mat, rotation):
    mat = mat.reshape(-1, 6, 6)
    accp = torch.tensor([[[0, 1, 0], [-1, 0, 0], [0, 0, 0]], [[0, 0, 1], [0, 0, 0], [-1, 0, 0]],
                        [[0, 0, 0], [0, 0, 1], [0, -1, 0]]], dtype=mat.dtype, device=mat.device)

    rotation_accp = torch.einsum('kab,nbc->knac', accp, rotation)
    rotation_6d = torch.cat((rotation[..., 0], rotation[..., 1]), dim=-1)
    rotation_6d_accp = torch.cat(
        (rotation_accp[..., 0], rotation_accp[..., 1]), dim=-1)

    rotation_t_6d = torch.einsum('nab,nb->na', mat, rotation_6d)
    rotation_t_6d_accp = torch.einsum('nab,knb->kna', mat, rotation_6d_accp)

    rotation_mat_0 = rotation_t_6d[..., :3]
    rotation_mat_0_accp = rotation_t_6d_accp[..., :3]
    t_rotation_mat_0, t_rotation_mat_0_accp = accp_normalize(
        rotation_mat_0, rotation_mat_0_accp)
    rotation_mat_1 = rotation_t_6d[..., 3:]
    rotation_mat_1_accp = rotation_t_6d_accp[..., 3:]
    dot = (t_rotation_mat_0 * rotation_mat_1).sum(dim=-1, keepdim=True)
    dot_accp = (t_rotation_mat_0_accp*rotation_mat_1 +
                t_rotation_mat_0*rotation_mat_1_accp).sum(dim=-1, keepdim=True)
    raw_t_mat_1 = rotation_mat_1 - dot*t_rotation_mat_0
    raw_t_mat_1_accp = rotation_mat_1_accp - \
        (dot_accp * t_rotation_mat_0 + dot * t_rotation_mat_0_accp)
    t_rotation_mat_1, t_rotation_mat_1_accp = accp_normalize(
        raw_t_mat_1, raw_t_mat_1_accp)
    t_rotation_mat_2, t_rotation_mat_2_accp = accp_cross(
        t_rotation_mat_0, t_rotation_mat_0_accp, t_rotation_mat_1, t_rotation_mat_1_accp)
    t_rotation = torch.cat([t_rotation_mat_0.unsqueeze(-1),
                           t_rotation_mat_1.unsqueeze(-1), t_rotation_mat_2.unsqueeze(-1)], dim=-1)
    t_rotation_accp = torch.cat([t_rotation_mat_0_accp.unsqueeze(
        -1), t_rotation_mat_1_accp.unsqueeze(-1), t_rotation_mat_2_accp.unsqueeze(-1)], dim=-1)
    delta = torch.einsum('knab,nbc->knac', t_rotation_accp,
                         t_rotation.transpose(-1, -2))
    vector = delta[..., [0, 0, 1], [1, 2, 2]]

    return t_rotation, vector.transpose(0, 1).det().abs().log()


class Condition36Trans(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.net = ConditionalTransform(feature_dim, 36)

    def forward(self, rotation, permute=None, feature=None):
        mat = self.net(feature).reshape(-1, 6, 6) + \
            torch.eye(6, device=rotation.device).unsqueeze(0)
        return calculate_36(mat, rotation)

    def inverse(self, rotation, permute=None, feature=None):
        mat = torch.linalg.inv(self.net(
            feature).reshape(-1, 6, 6)+torch.eye(6, device=rotation.device).unsqueeze(0))
        return calculate_36(mat, rotation)


class Uncondition36Trans(nn.Module):
    def __init__(self):
        super().__init__()
        self.mat = nn.Parameter(torch.eye(6)+torch.randn(6, 6)*1e-3)

    def forward(self, rotation, permute=None, feature=None):
        mat = self.mat
        return calculate_36(mat, rotation)

    def inverse(self, rotation, permute=None, feature=None):
        mat = torch.linalg.inv(self.mat)
        return calculate_36(mat, rotation)
