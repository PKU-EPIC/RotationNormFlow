from nflows.distributions import Distribution
from scipy.integrate import solve_ivp
from scipy.special import i0
from pytorch3d.transforms import (
    matrix_to_quaternion,
    quaternion_to_matrix,
    random_rotations,
)
import torch
import numpy as np

'''class for matrix fisher'''

def quat_to_rotmat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:,
                                            1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack(
        [
            w2 + x2 - y2 - z2,
            2 * xy - 2 * wz,
            2 * wy + 2 * xz,
            2 * wz + 2 * xy,
            w2 - x2 + y2 - z2,
            2 * yz - 2 * wx,
            2 * xz - 2 * wy,
            2 * wx + 2 * yz,
            w2 - x2 - y2 + z2,
        ],
        dim=1,
    ).view(B, 3, 3)
    return rotMat

def proper_svd(A, clone=False):
    U, S, V = torch.svd(A)
    detU, detV = torch.det(U), torch.det(V)
    if clone:
        #print("run clone")
        U_clone = U.clone()
        V_clone = V.clone()
        S_clone = S.clone()
        U_clone[:, 2] *= detU
        S_clone[2] *= detU * detV
        V_clone[:, 2] *= detV
        return U_clone, S_clone, V_clone
    else:
        U[:, 2] *= detU
        S[2] *= detU * detV
        V[:, 2] *= detV
        return U, S, V


def proper_svd_N(A):
    U, S, V = torch.svd(A)
    detU, detV = torch.det(U), torch.det(V)
    U_clone = U.clone()
    V_clone = V.clone()
    S_clone = S.clone()
    U_clone[:, :, 2] *= detU.reshape(-1, 1)
    S_clone[:, 2] *= detU * detV
    V_clone[:, :, 2] *= detV.reshape(-1, 1)
    return U_clone, S_clone, V_clone


def matrix_fisher_norm_N(A, type_approx=0, approx_num=17890714):
    """
    the approximated normalizing constant for the matrix Fisher distribution
    A: (N, 3, 3)
    S: proper singular value of shape (N, 3,)
    type_approx:
        0 - approximation by almost uniform distribuitons when s is small
        1 - approximaiton by highly concentraed distributions when s_i+s_j is large
    """
    U, S, V = proper_svd_N(A)  # (N, 3, 3)
    if type_approx == 0:
        norm = 1.0 + 1.0 / 6.0 * (S**2).sum() + 1.0 / \
            6.0 * S[:, 0] * S[:, 1] * S[:, 2]
        norm = norm / torch.sum(S, dim=-1).exp()
    elif type_approx == 1:
        norm = 1.0 / torch.sqrt(
            8 * torch.pi * (S[:, 0] + S[:, 1]) *
            (S[:, 2] + S[:, 1]) * (S[:, 0] + S[:, 2])
        )
    elif type_approx == 2:
        random_rot = random_rotations(approx_num)
        trace = (random_rot * A).sum(dim=-1).sum(dim=-1)
        norm = (trace - S.sum()).exp().mean()
    elif type_approx == 3:

        def func(t, y):
            z1 = 1 / 2 * (S[0] - S[1]) * (1 - t)
            b1 = i0(z1) * np.exp(-np.abs(z1))
            z2 = 1 / 2 * (S[0] + S[1]) * (1 + t)
            b2 = i0(z2) * np.exp(-np.abs(z2))
            return b1 * b2 * np.exp((t - 1) * (S[1] + S[2])) / 2

        sol = solve_ivp(func, [-1, 1], np.array([0]), max_step=0.01)
        return torch.from_numpy(sol.y[:, -1])
    else:
        return NotImplementedError()
    return norm

def sample_bingham(
    A, num_samples, Omega, Gaussian_std, b, M_star, oversampling_ratio=8
):
    """
    Sampling from a Bingham distribution with 4x4 matrix parameter A.
    Here we assume that A is a diagonal matrix (needed for matrix-Fisher sampling).
    Bing(A) is simulated by rejection sampling from ACG(I + 2A/b) (since ACG > Bingham everywhere).
    Rejection sampling is batched + differentiable (using re-parameterisation trick).
    For further details, see: https://arxiv.org/pdf/1310.8110.pdf and
    https://github.com/tylee-fdcl/Matrix-Fisher-Distribution
    :param A: (4,) tensor parameter of Bingham distribution on 3-sphere.
        Represents the diagonal of a 4x4 diagonal matrix.
    :param num_samples: scalar. Number of samples to draw.
    :param Omega: (4,) Optional tensor parameter of ACG distribution on 3-sphere.
    :param Gaussian_std: (4,) Optional tensor parameter (standard deviations) of diagonal Gaussian in R^4.
    :param num_samples:
    :param b: Hyperparameter for rejection sampling using envelope ACG distribution with
        Omega = I + 2A/b
    :param oversampling_ratio: scalar. To make rejection sampling batched, we sample num_samples * oversampling_ratio,
        then reject samples according to rejection criterion, and hopeffully the number of samples remaining is
        > num_samples.
    :return: samples: (num_samples, 4) and accept_ratio
    """
    samples_obtained = False
    while not samples_obtained:
        eps = torch.randn(num_samples * oversampling_ratio,
                          4, device=A.device).float()
        y = Gaussian_std * eps
        samples = y / torch.norm(y, dim=1, keepdim=True)

        p_Bing_star = torch.exp(
            -torch.einsum("bn,n,bn->b", samples, A, samples)
        )  # (num_samples * oversampling_ratio,)
        p_ACG_star = torch.einsum("bn,n,bn->b", samples, Omega, samples) ** (
            -2
        )  # (num_samples * oversampling_ratio,)
        # assert torch.all(p_Bing_star <= M_star * p_ACG_star + 1e-6)

        w = torch.rand(num_samples * oversampling_ratio, device=A.device)
        accept_vector = w < p_Bing_star / (
            M_star * p_ACG_star
        )  # (num_samples * oversampling_ratio,)
        num_accepted = accept_vector.sum().item()
        if num_accepted >= num_samples:
            samples = samples[accept_vector, :]  # (num_accepted, 4)
            samples = samples[:num_samples, :]  # (num_samples, 4)
            samples_obtained = True
            accept_ratio = num_accepted / num_samples * 4
        else:
            print(
                "Failed sampling. {} samples accepted, {} samples required.".format(
                    num_accepted, num_samples
                )
            )

    return samples, accept_ratio


def sample_matrix_fisher(A, num_samples, b=1.5, oversampling_ratio=8):
    """
    :param A: (3, 3)
    :param b: Hyperparameter for rejection sampling using envelope ACG distribution.
    """
    U, S, V = proper_svd(A)

    bingham_A = torch.zeros(4)
    bingham_A[1] = 2 * (S[1] + S[2])
    bingham_A[2] = 2 * (S[0] + S[2])
    bingham_A[3] = 2 * (S[0] + S[1])

    Omega = torch.ones(4) + 2 * bingham_A / b
    Gaussian_std = Omega ** (-0.5)
    M_star = np.exp(-(4 - b) / 2) * (
        (4 / b) ** 2
    )  # Bound for rejection sampling: Bing(A) <= Mstar(b)ACG(I+2A/b)

    quat_samples, accept_ratio = sample_bingham(
        A=bingham_A,
        num_samples=num_samples,
        Omega=Omega,
        Gaussian_std=Gaussian_std,
        b=b,
        M_star=M_star,
        oversampling_ratio=oversampling_ratio,
    )
    pose_R_samples_batch = quat_to_rotmat(quat=quat_samples.view(-1, 4)).view(
        num_samples, 3, 3
    )
    pose_R_samples_batch = U @ pose_R_samples_batch.to(U.device) @ V.T

    return pose_R_samples_batch

class MatrixFisherN(Distribution):
    def __init__(self, A, norm_type=1, approx_num=None):
        '''A: (N, 3, 3)'''
        super().__init__()
        self.A = A  # (N, 3, 3)
        norm_type = norm_type
        self.norm = matrix_fisher_norm_N(A, norm_type, approx_num)  # (N)

    def _log_prob(self, inputs, context=9):
        if inputs.device != self.A.device:
            self.A = self.A.to(inputs.device)
            self.norm = self.norm.to(inputs.device)
        if inputs.shape[-1] == 4:
            inputs_9d = quaternion_to_matrix(inputs)
        elif inputs.shape[-1] == 3 and inputs.shape[-2] == 3:
            inputs_9d = inputs

        inputs_9d = inputs_9d.reshape(self.A.shape[0], -1, 3, 3)
        U, S, V = proper_svd_N(self.A)
        trace = (inputs_9d * self.A.reshape(-1, 1, 3, 3)
                 ).sum(dim=-1).sum(dim=-1)
        result = (trace - torch.sum(S, dim=-1).reshape(-1, 1)) - \
            self.norm.log().reshape(-1, 1)
        return result.reshape(-1)

    def _sample(self, num_samples, context=9):
        result = torch.empty(
            (self.A.size(0), num_samples, 3, 3), device=self.A.device)
        for i in range(self.A.size(0)):
            result[i] = sample_matrix_fisher(
                self.A[i], num_samples=num_samples)
        if context == 9:
            return result
        elif context == 4:
            return matrix_to_quaternion(result)