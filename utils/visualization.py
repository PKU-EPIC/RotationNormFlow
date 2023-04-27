import torch
import pytorch3d.transforms as pttf
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
#from utils import rotation_target
from matplotlib.markers import MarkerStyle
import utils.sd as sd
import os
import cv2

all_pascal_classes = [
    "aeroplane",
    "bicycle",
    "boat",
    "bottle",
    "bus",
    "car",
    "chair",
    "diningtable",
    "motorbike",
    "sofa",
    "train",
    "tvmonitor",
]
all_modelnet_classes = ['bathtub', 'bed', 'chair', 'desk', 'dresser',
                        'monitor', 'night_stand', 'sofa', 'table', 'toilet']
symsol1 = ['cone', 'cube', 'cyl', 'icosa', 'tet', 'sphereX', 'cylO',  'tetX']


def visualize_sample_nofn(config, rotations, log_prob, category, gt=None, idx=0, img=None, draw_list=None):
    rotations = rotations.cpu()
    log_prob = log_prob.cpu()
    if gt is not None:
        gt = gt.cpu()
    #pro = torch.exp(ldjs)

    #norm = pro.mean()
    # print(norm)
    can_rotation = torch.Tensor([[0.6226537, -0.27743655, -0.7316634],
                                 [0.73486626, -0.1139197,  0.66857624],
                                 [-0.26883835, -0.9539662,  0.13294627]])

    fig = visualize_so3_probabilities_sample(rotations,
                                             torch.zeros_like(log_prob),
                                             ax=None,
                                             fig=None,
                                             rotations_gt=gt,
                                             display_threshold_probability=-1000,
                                             scatter_size=config.scatter_size,
                                             )
    if config.dataset == 'pascal3d':
        save_path = os.path.join(
            "exps_pascal3d", config.exp_name, config.date, "draw", all_pascal_classes[category.item()])
    elif 'symsol' in config.dataset:
        save_path = os.path.join(
            "exps_symsol", config.exp_name, config.date, "draw", symsol1[category.item()])
    elif 'modelnet' in config.dataset:
        save_path = os.path.join(
            "exps_modelnet", config.exp_name, config.date, "draw", all_modelnet_classes[category.item()])
    else:
        raise NotImplementedError()
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    fig.suptitle(save_path)
    final_img = os.path.join(save_path, f"result_f_{idx}.jpg")
    #print(f'{final_img}\n gt:\n {gt}\n')
    fig.savefig(final_img)
    np.save(os.path.join(
        save_path, f"matrix_{idx}.npy"), rotations.detach().numpy())

    img_array = []
    detail_save_path = os.path.join(save_path, f"detail_{idx}")
    if not os.path.exists(detail_save_path):
        os.makedirs(detail_save_path)
    for (layer_num, layer_name, layer_rotation) in draw_list:
        layer_rotation = layer_rotation.cpu()
        fig = visualize_so3_probabilities_sample(layer_rotation,
                                                 torch.zeros_like(log_prob),
                                                 ax=None,
                                                 fig=None,
                                                 display_threshold_probability=-1000,
                                                 scatter_size=config.scatter_size,
                                                 )
        layer_title = f'layer num {layer_num}, {layer_name}'
        fig.suptitle(layer_title)
        img_name = os.path.join(detail_save_path,
                                f"{str(layer_num).zfill(2)}.jpg")
        fig.savefig(img_name)
        img_array.append(cv2.imread(img_name))
        np.save(os.path.join(detail_save_path,
                f"{str(layer_num).zfill(2)}.npy"), layer_rotation)
    if len(img_array) > 0:
        img_array.append(cv2.imread(final_img))
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        videowrite = cv2.VideoWriter(os.path.join(
            save_path, f"video_{idx}.mp4"), fourcc, 2, (1600, 800))
        for draw_img in img_array:
            videowrite.write(draw_img)
        videowrite.release()
        cv2.destroyAllWindows()

    if img is not None:
        picture_path = os.path.join(save_path, f"picture_{idx}.jpg")
        cv2.imwrite(picture_path, img.permute(
            1, 2, 0).detach().cpu().numpy()*255)
        picture = cv2.imread(picture_path)

        def to_2d(start, vector_3d, length):
            delta = (int(vector_3d[0]*length), int(-vector_3d[1]*length))
            return start[0]+delta[0], start[1]+delta[1]

        reshape_k = 2
        new_size = 224*reshape_k
        picture = cv2.resize(picture, dsize=None, fx=reshape_k,
                             fy=reshape_k, interpolation=cv2.INTER_LINEAR)

        for i in range(min(100, rotations.shape[0])):
            est = rotations[i]
            cv2.arrowedLine(picture, (new_size//2, new_size//2), to_2d(
                (new_size//2, new_size//2), est[0], new_size*0.3), color=(200, 0, 0), thickness=2)
            cv2.arrowedLine(picture, (new_size//2, new_size//2), to_2d(
                (new_size//2, new_size//2), est[1], new_size*0.3), color=(0, 200, 0), thickness=2)
            cv2.arrowedLine(picture, (new_size//2, new_size//2), to_2d(
                (new_size//2, new_size//2), est[2], new_size*0.3), color=(0, 0, 200), thickness=2)

        cv2.arrowedLine(picture, (new_size//2, new_size//2), to_2d(
            (new_size//2, new_size//2), gt[0], new_size*0.4), color=(255, 0, 0), thickness=4)
        cv2.arrowedLine(picture, (new_size//2, new_size//2), to_2d(
            (new_size//2, new_size//2), gt[1], new_size*0.4), color=(0, 255, 0), thickness=4)
        cv2.arrowedLine(picture, (new_size//2, new_size//2), to_2d(
            (new_size//2, new_size//2), gt[2], new_size*0.4), color=(0, 0, 255), thickness=4)
        cv2.imwrite(picture_path, picture)
    plt.clf()

    fig = plt.figure()
    ax1 = plt.axes(projection='3d')
    xyz = pttf.matrix_to_axis_angle(rotations)
    ax1.scatter3D(xyz[:, 0], xyz[:, 1], xyz[:, 2])
    fig.savefig(os.path.join(save_path, f"result_aa_{idx}.jpg"))
    plt.clf()

    #norm = pro.mean()
    # print(norm)
    can_rotation = torch.Tensor([[0.6226537, -0.27743655, -0.7316634],
                                 [0.73486626, -0.1139197,  0.66857624],
                                 [-0.26883835, -0.9539662,  0.13294627]])

    fig = visualize_so3_probabilities_sample(rotations,
                                             log_prob,
                                             ax=None,
                                             fig=None,
                                             rotations_gt=gt,
                                             display_threshold_probability=-1000,
                                             scatter_size=config.scatter_size,
                                             canonical_rotation=can_rotation,
                                             )
    if config.dataset == 'pascal3d':
        save_path = os.path.join(
            "exps_pascal3d", config.exp_name, config.date, all_pascal_classes[category.item()])
    elif 'symsol' in config.dataset:
        save_path = os.path.join(
            "exps_symsol", config.exp_name, config.date, symsol1[category.item()])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fig.suptitle(save_path)
    fig.show()
    fig.savefig(os.path.join(save_path, f"result_{idx}.jpg"))
    if img is not None:
        cv2.imwrite(os.path.join(save_path, f"picture_{idx}.jpg"), img.permute(
            1, 2, 0).detach().cpu().numpy()*255)
    plt.clf()


def visualize_so3_probabilities(rotations,
                                probabilities,
                                rotations_gt=None,
                                draw_mode=False,
                                ax=None,
                                fig=None,
                                display_threshold_probability=0,
                                show_color_wheel=True,
                                canonical_rotation=np.eye(3),
                                scatter_size=4e3,):
    """Plot a single distribution on SO(3) using the tilt-colored method.

    Args:
      rotations: [N, 3, 3] tensor of rotation matrices
      probabilities: [N] tensor of probabilities
      rotations_gt: [N_gt, 3, 3] or [3, 3] ground truth rotation matrices
      ax: The matplotlib.pyplot.axis object to paint
      fig: The matplotlib.pyplot.figure object to paint
      display_threshold_probability: The probability threshold below which to omit
        the marker
      to_image: If True, return a tensor containing the pixels of the finished
        figure; if False return the figure itself
      show_color_wheel: If True, display the explanatory color wheel which matches
        color on the plot with tilt angle
      canonical_rotation: A [3, 3] rotation matrix representing the 'display
        rotation', to change the view of the distribution.  It rotates the
        canonical axes so that the view of SO(3) on the plot is different, which
        can help obtain a more informative view.

    Returns:
      A matplotlib.pyplot.figure object, or a tensor of pixels if to_image=True.
    """

    def _show_single_marker(ax, rotation, marker, edgecolors=True,
                            facecolors=False):
        eulers = pttf.matrix_to_euler_angles(
            rotation, 'ZYX')[..., [-1, -2, -3]]
        xyz = rotation[:, 0]
        tilt_angle = eulers[0]
        longitude = np.arctan2(xyz[0], -xyz[1])
        latitude = np.arcsin(xyz[2])

        color = cmap(0.5 + tilt_angle.item() / 2 / np.pi)
        ax.scatter(longitude, latitude, s=1000,
                   edgecolors=color if edgecolors else 'none',
                   facecolors=facecolors if facecolors else 'none',
                   marker=marker,
                   linewidth=2)

    if ax is None:
        fig = plt.figure(figsize=(8, 4), dpi=200)
        fig.set_tight_layout(True)
        ax = fig.add_subplot(111, projection='mollweide')
    if rotations_gt is not None and len(rotations_gt.shape) == 2:
        rotations_gt = rotations_gt[None]

    display_rotations = rotations @ canonical_rotation
    cmap = plt.cm.hsv
    scatterpoint_scaling = scatter_size
    eulers_queries = pttf.matrix_to_euler_angles(
        display_rotations, 'ZYX')[..., [-1, -2, -3]]
    xyz = display_rotations[:, :, 0]
    tilt_angles = eulers_queries[:, 0]

    longitudes = np.arctan2(xyz[:, 0], -xyz[:, 1])
    latitudes = np.arcsin(xyz[:, 2])

    if hasattr(display_threshold_probability, '__len__') and len(display_threshold_probability) == 2:
        which_to_display = (display_threshold_probability[0] < probabilities) & (
            probabilities < display_threshold_probability[1])
    else:
        which_to_display = (probabilities > display_threshold_probability)

    print(torch.sum(which_to_display))

    if rotations_gt is not None:
        # The visualization is more comprehensible if the GT
        # rotation markers are behind the output with white filling the interior.
        display_rotations_gt = rotations_gt @ canonical_rotation

        for rotation in display_rotations_gt:
            _show_single_marker(ax, rotation, 'o')
        # Cover up the centers with white markers
        for rotation in display_rotations_gt:
            _show_single_marker(ax, rotation, 'o', edgecolors=False,)

    if draw_mode:
        # Display the mode of the distribution
        rotations_mode = rotations[probabilities.argmax()]
        display_rotations_mode = rotations_mode @ canonical_rotation
        _show_single_marker(ax, display_rotations_mode, 'o')
        # Cover up the centers with white markers
        _show_single_marker(ax, display_rotations_mode, 'o', edgecolors=False,)

    # Display the distribution
    ax.scatter(
        longitudes[which_to_display],
        latitudes[which_to_display],
        s=scatterpoint_scaling * probabilities[which_to_display],
        c=cmap(0.5 + tilt_angles[which_to_display] / 2. / np.pi))

    # coord = np.stack((longitudes, latitudes, tilt_angles.numpy()), axis=-1)
    # coord = np.sort(coord, axis=0)
    # print(coord.shape)
    # print(coord)

    ax.grid()
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if show_color_wheel:
        # Add a color wheel showing the tilt angle to color conversion.
        ax = fig.add_axes([0.86, 0.17, 0.12, 0.12], projection='polar')
        theta = np.linspace(-3 * np.pi / 2, np.pi / 2, 200)
        radii = np.linspace(0.4, 0.5, 2)
        _, theta_grid = np.meshgrid(radii, theta)
        colormap_val = 0.5 + theta_grid / np.pi / 2.
        ax.pcolormesh(theta, radii, colormap_val.T, cmap=cmap)
        ax.set_yticklabels([])
        ax.set_xticklabels([r'90$\degree$', None,
                            r'180$\degree$', None,
                            r'270$\degree$', None,
                            r'0$\degree$'], fontsize=14)
        ax.spines['polar'].set_visible(False)
        plt.text(0.5, 0.5, 'Tilt', fontsize=14,
                 horizontalalignment='center',
                 verticalalignment='center', transform=ax.transAxes)

    return fig


def visualize_so3_probabilities_sample(rotations,
                                       probabilities,
                                       rotations_gt=None,
                                       draw_mode=False,
                                       ax=None,
                                       fig=None,
                                       display_threshold_probability=0,
                                       show_color_wheel=True,
                                       canonical_rotation=np.eye(3),
                                       scatter_size=4e3,):
    """Plot a single distribution on SO(3) using the tilt-colored method.

    Args:
      rotations: [N, 3, 3] tensor of rotation matrices
      probabilities: [N] tensor of probabilities
      rotations_gt: [N_gt, 3, 3] or [3, 3] ground truth rotation matrices
      ax: The matplotlib.pyplot.axis object to paint
      fig: The matplotlib.pyplot.figure object to paint
      display_threshold_probability: The probability threshold below which to omit
        the marker
      to_image: If True, return a tensor containing the pixels of the finished
        figure; if False return the figure itself
      show_color_wheel: If True, display the explanatory color wheel which matches
        color on the plot with tilt angle
      canonical_rotation: A [3, 3] rotation matrix representing the 'display
        rotation', to change the view of the distribution.  It rotates the
        canonical axes so that the view of SO(3) on the plot is different, which
        can help obtain a more informative view.

    Returns:
      A matplotlib.pyplot.figure object, or a tensor of pixels if to_image=True.
    """

    def _show_single_marker(ax, rotation, marker, edgecolors=True,
                            facecolors=False):
        eulers = pttf.matrix_to_euler_angles(
            rotation, 'ZYX')[..., [-1, -2, -3]]
        xyz = rotation[:, 0]
        tilt_angle = eulers[0]
        longitude = np.arctan2(xyz[0], -xyz[1])
        latitude = np.arcsin(xyz[2])

        color = cmap(0.5 + tilt_angle.item() / 2 / np.pi)
        ax.scatter(longitude, latitude, s=1000,
                   edgecolors=color if edgecolors else 'none',
                   facecolors=facecolors if facecolors else 'none',
                   marker=marker,
                   linewidth=2)

    if ax is None:
        fig = plt.figure(figsize=(8, 4), dpi=200)
        fig.set_tight_layout(True)
        ax = fig.add_subplot(111, projection='mollweide')
    if rotations_gt is not None and len(rotations_gt.shape) == 2:
        rotations_gt = rotations_gt[None]

    display_rotations = rotations @ canonical_rotation
    cmap = plt.cm.hsv
    scatterpoint_scaling = scatter_size
    eulers_queries = pttf.matrix_to_euler_angles(
        display_rotations, 'ZYX')[..., [-1, -2, -3]]
    xyz = display_rotations[:, :, 0]
    tilt_angles = eulers_queries[:, 0]

    longitudes = np.arctan2(xyz[:, 0], -xyz[:, 1])
    latitudes = np.arcsin(xyz[:, 2])

    if hasattr(display_threshold_probability, '__len__') and len(display_threshold_probability) == 2:
        which_to_display = (display_threshold_probability[0] < probabilities) & (
            probabilities < display_threshold_probability[1])
    else:
        which_to_display = (probabilities > display_threshold_probability)

    print(torch.sum(which_to_display))

    if rotations_gt is not None:
        # The visualization is more comprehensible if the GT
        # rotation markers are behind the output with white filling the interior.
        display_rotations_gt = rotations_gt @ canonical_rotation

        for rotation in display_rotations_gt:
            _show_single_marker(ax, rotation, 'o')
        # Cover up the centers with white markers
        for rotation in display_rotations_gt:
            _show_single_marker(ax, rotation, 'o', edgecolors=False,)

    if draw_mode:
        # Display the mode of the distribution
        rotations_mode = rotations[probabilities.argmax()]
        display_rotations_mode = rotations_mode @ canonical_rotation
        _show_single_marker(ax, display_rotations_mode, 'o')
        # Cover up the centers with white markers
        _show_single_marker(ax, display_rotations_mode, 'o', edgecolors=False,)

    # Display the distribution
    ax.scatter(
        longitudes[which_to_display],
        latitudes[which_to_display],
        s=scatterpoint_scaling,
        c=cmap(0.5 + tilt_angles[which_to_display] / 2. / np.pi))

    # coord = np.stack((longitudes, latitudes, tilt_angles.numpy()), axis=-1)
    # coord = np.sort(coord, axis=0)
    # print(coord.shape)
    # print(coord)

    ax.grid()
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if show_color_wheel:
        # Add a color wheel showing the tilt angle to color conversion.
        ax = fig.add_axes([0.86, 0.17, 0.12, 0.12], projection='polar')
        theta = np.linspace(-3 * np.pi / 2, np.pi / 2, 200)
        radii = np.linspace(0.4, 0.5, 2)
        _, theta_grid = np.meshgrid(radii, theta)
        colormap_val = 0.5 + theta_grid / np.pi / 2.
        ax.pcolormesh(theta, radii, colormap_val.T, cmap=cmap)
        ax.set_yticklabels([])
        ax.set_xticklabels([r'90$\degree$', None,
                            r'180$\degree$', None,
                            r'270$\degree$', None,
                            r'0$\degree$'], fontsize=14)
        ax.spines['polar'].set_visible(False)
        plt.text(0.5, 0.5, 'Tilt', fontsize=14,
                 horizontalalignment='center',
                 verticalalignment='center', transform=ax.transAxes)

    return fig
