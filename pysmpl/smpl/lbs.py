# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

import torch
import torch.nn.functional as F
import roma
import math


def rot_mat_to_euler(rot_mats):
    # Calculates rotation matrix to euler angles
    # Careful for extreme cases of eular angles like [0.0, pi, 0.0]

    sy = torch.sqrt(rot_mats[:, 0, 0] * rot_mats[:, 0, 0] +
                    rot_mats[:, 1, 0] * rot_mats[:, 1, 0])
    return torch.atan2(-rot_mats[:, 2, 0], sy)


def find_dynamic_lmk_idx_and_bcoords(vertices, pose, dynamic_lmk_faces_idx,
                                     dynamic_lmk_b_coords,
                                     neck_kin_chain, dtype=torch.float32):
    ''' Compute the faces, barycentric coordinates for the dynamic landmarks


        To do so, we first compute the rotation of the neck around the y-axis
        and then use a pre-computed look-up table to find the faces and the
        barycentric coordinates that will be used.

        Special thanks to Soubhik Sanyal (soubhik.sanyal@tuebingen.mpg.de)
        for providing the original TensorFlow implementation and for the LUT.

        Parameters
        ----------
        vertices: torch.tensor BxVx3, dtype = torch.float32
            The tensor of input vertices
        pose: torch.tensor Bx(Jx3), dtype = torch.float32
            The current pose of the body model
        dynamic_lmk_faces_idx: torch.tensor L, dtype = torch.long
            The look-up table from neck rotation to faces
        dynamic_lmk_b_coords: torch.tensor Lx3, dtype = torch.float32
            The look-up table from neck rotation to barycentric coordinates
        neck_kin_chain: list
            A python list that contains the indices of the joints that form the
            kinematic chain of the neck.
        dtype: torch.dtype, optional

        Returns
        -------
        dyn_lmk_faces_idx: torch.tensor, dtype = torch.long
            A tensor of size BxL that contains the indices of the faces that
            will be used to compute the current dynamic landmarks.
        dyn_lmk_b_coords: torch.tensor, dtype = torch.float32
            A tensor of size BxL that contains the indices of the faces that
            will be used to compute the current dynamic landmarks.
    '''

    batch_size = vertices.shape[0]

    aa_pose = torch.index_select(pose.view(batch_size, -1, 3), 1,
                                 neck_kin_chain)
    rot_mats = batch_rodrigues(
        aa_pose.view(-1, 3), dtype=dtype).view(batch_size, -1, 3, 3)

    rel_rot_mat = torch.eye(
        3, device=vertices.device, dtype=dtype).unsqueeze_(dim=0).repeat(
        batch_size, 1, 1)
    for idx in range(len(neck_kin_chain)):
        rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)

    y_rot_angle = torch.round(
        torch.clamp(-rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi,
                    max=39)).to(dtype=torch.long)
    neg_mask = y_rot_angle.lt(0).to(dtype=torch.long)
    mask = y_rot_angle.lt(-39).to(dtype=torch.long)
    neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
    y_rot_angle = (neg_mask * neg_vals +
                   (1 - neg_mask) * y_rot_angle)

    dyn_lmk_faces_idx = torch.index_select(dynamic_lmk_faces_idx,
                                           0, y_rot_angle)
    dyn_lmk_b_coords = torch.index_select(dynamic_lmk_b_coords,
                                          0, y_rot_angle)

    return dyn_lmk_faces_idx, dyn_lmk_b_coords


def vertices2landmarks(vertices, faces, lmk_faces_idx, lmk_bary_coords):
    ''' Calculates landmarks by barycentric interpolation

        Parameters
        ----------
        vertices: torch.tensor BxVx3, dtype = torch.float32
            The tensor of input vertices
        faces: torch.tensor Fx3, dtype = torch.long
            The faces of the mesh
        lmk_faces_idx: torch.tensor L, dtype = torch.long
            The tensor with the indices of the faces used to calculate the
            landmarks.
        lmk_bary_coords: torch.tensor Lx3, dtype = torch.float32
            The tensor of barycentric coordinates that are used to interpolate
            the landmarks

        Returns
        -------
        landmarks: torch.tensor BxLx3, dtype = torch.float32
            The coordinates of the landmarks for each mesh in the batch
    '''
    # Extract the indices of the vertices for each face
    # BxLx3
    batch_size, num_verts = vertices.shape[:2]
    device = vertices.device

    lmk_faces = torch.index_select(faces, 0, lmk_faces_idx.view(-1)).view(
        batch_size, -1, 3)

    lmk_faces += torch.arange(
        batch_size, dtype=torch.long, device=device).view(-1, 1, 1) * num_verts

    lmk_vertices = vertices.view(-1, 3)[lmk_faces].view(
        batch_size, -1, 3, 3)

    landmarks = torch.einsum('blfi,blf->bli', [lmk_vertices, lmk_bary_coords])
    return landmarks


def joints2bones(joints, parents):
    ''' Decompose joints location to bone length and direction.

        Parameters
        ----------
        joints: torch.tensor Bx24x3
    '''
    assert joints.shape[1] == parents.shape[0]
    bone_dirs = torch.zeros_like(joints)
    bone_lens = torch.zeros_like(joints[:, :, :1])

    for c_id in range(parents.shape[0]):
        p_id = parents[c_id]
        if p_id == -1:
            # Parent node
            bone_dirs[:, c_id] = joints[:, c_id]
        else:
            # Child node
            # (B, 3)
            diff = joints[:, c_id] - joints[:, p_id]
            length = torch.norm(diff, dim=1, keepdim=True) + 1e-8
            direct = diff / length

            bone_dirs[:, c_id] = direct
            bone_lens[:, c_id] = length

    return bone_dirs, bone_lens


def bones2joints(bone_dirs, bone_lens, parents):
    ''' Recover bone length and direction to joints location.

        Parameters
        ----------
        bone_dirs: torch.tensor 1x24x3
        bone_lens: torch.tensor Bx24x1
    '''
    batch_size = bone_lens.shape[0]
    joints = torch.zeros_like(bone_dirs).expand(batch_size, 24, 3)

    for c_id in range(parents.shape[0]):
        p_id = parents[c_id]
        if p_id == -1:
            # Parent node
            joints[:, c_id] = bone_dirs[:, c_id]
        else:
            # Child node
            joints[:, c_id] = joints[:, p_id] + bone_dirs[:, c_id] * bone_lens[:, c_id]

    return joints


def lbs(betas, pose, v_template, shapedirs, posedirs, J_regressor, J_regressor_h36m, parents,
        lbs_weights, pose2rot=True, dtype=torch.float32):
    ''' Performs Linear Blend Skinning with the given shape and pose parameters

        Parameters
        ----------
        betas : torch.tensor BxNB
            The tensor of shape parameters
        pose : torch.tensor Bx(J + 1) * 3
            The pose parameters in axis-angle format
        v_template torch.tensor BxVx3
            The template mesh that will be deformed
        shapedirs : torch.tensor 1xNB
            The tensor of PCA shape displacements
        posedirs : torch.tensor Px(V * 3)
            The pose PCA coefficients
        J_regressor : torch.tensor JxV
            The regressor array that is used to calculate the joints from
            the position of the vertices
        parents: torch.tensor J
            The array that describes the kinematic tree for the model
        lbs_weights: torch.tensor N x V x (J + 1)
            The linear blend skinning weights that represent how much the
            rotation matrix of each part affects each vertex
        pose2rot: bool, optional
            Flag on whether to convert the input pose tensor to rotation
            matrices. The default value is True. If False, then the pose tensor
            should already contain rotation matrices and have a size of
            Bx(J + 1)x9
        dtype: torch.dtype, optional

        Returns
        -------
        verts: torch.tensor BxVx3
            The vertices of the mesh after applying the shape and pose
            displacements.
        joints: torch.tensor BxJx3
            The joints of the model
        rot_mats: torch.tensor BxJx3x3
            The rotation matrics of each joints
    '''
    batch_size = max(betas.shape[0], pose.shape[0])
    device = betas.device

    # Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    # Get the joints
    # NxJx3 array
    J = vertices2joints(J_regressor, v_shaped)

    # 3. Add pose blend shapes
    # N x J x 3 x 3
    ident = torch.eye(3, dtype=dtype, device=device)
    if pose2rot:
        if pose.numel() == batch_size * 24 * 4:
            rot_mats = quat_to_rotmat(pose.reshape(batch_size * 24, 4)).reshape(batch_size, 24, 3, 3)
        else:
            rot_mats = batch_rodrigues(
                pose.view(-1, 3), dtype=dtype).view([batch_size, -1, 3, 3])

        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        # (N x P) x (P, V * 3) -> N x V x 3
        pose_offsets = torch.matmul(pose_feature, posedirs) \
            .view(batch_size, -1, 3)
    else:
        pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
        rot_mats = pose.view(batch_size, -1, 3, 3)

        pose_offsets = torch.matmul(pose_feature.view(batch_size, -1),
                                    posedirs).view(batch_size, -1, 3)

    v_posed = pose_offsets + v_shaped
    # 4. Get the global joint location
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents[:24], dtype=dtype)

    # 5. Do skinning:
    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                               dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]

    J_from_verts = vertices2joints(J_regressor_h36m, verts)

    return verts, J_transformed, rot_mats, J_from_verts


def hybrik(betas, global_orient, pose_skeleton, phis,
           v_template, shapedirs, posedirs, J_regressor, J_regressor_h36m, parents, children,
           lbs_weights, dtype=torch.float32, train=False, leaf_thetas=None):
    ''' Performs Linear Blend Skinning with the given shape and skeleton joints

        Parameters
        ----------
        betas : torch.tensor BxNB
            The tensor of shape parameters
        global_orient : torch.tensor Bx3
            The tensor of global orientation
        pose_skeleton : torch.tensor BxJ*3
            The pose skeleton in (X, Y, Z) format
        phis : torch.tensor BxJx2
            The rotation on bone axis parameters
        v_template torch.tensor BxVx3
            The template mesh that will be deformed
        shapedirs : torch.tensor 1xNB
            The tensor of PCA shape displacements
        posedirs : torch.tensor Px(V * 3)
            The pose PCA coefficients
        J_regressor : torch.tensor JxV
            The regressor array that is used to calculate the joints from
            the position of the vertices
        J_regressor_h36m : torch.tensor 17xV
            The regressor array that is used to calculate the 17 Human3.6M joints from
            the position of the vertices
        parents: torch.tensor J
            The array that describes the kinematic parents for the model
        children: dict
            The dictionary that describes the kinematic chidrens for the model
        lbs_weights: torch.tensor N x V x (J + 1)
            The linear blend skinning weights that represent how much the
            rotation matrix of each part affects each vertex
        dtype: torch.dtype, optional

        Returns
        -------
        verts: torch.tensor BxVx3
            The vertices of the mesh after applying the shape and pose
            displacements.
        joints: torch.tensor BxJx3
            The joints of the model
        rot_mats: torch.tensor BxJx3x3
            The rotation matrics of each joints
    '''
    batch_size = max(betas.shape[0], pose_skeleton.shape[0])
    device = betas.device

    # 1. Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    # 2. Get the rest joints
    # NxJx3 array
    rest_J = vertices2joints(J_regressor, v_shaped)
    # if leaf_thetas is not None:
    #     rest_J = vertices2joints(J_regressor, v_shaped)
    # else:
    #     rest_J = torch.zeros((v_shaped.shape[0], 29, 3), dtype=dtype, device=device)
    #     rest_J[:, :24] = vertices2joints(J_regressor, v_shaped)
    #
    #     leaf_number = [411, 2445, 5905, 3216, 6617]
    #     leaf_vertices = v_shaped[:, leaf_number].clone()
    #     rest_J[:, 24:] = leaf_vertices

    # 3. Get the rotation matrics
    rot_mats, rotate_rest_pose = batch_inverse_kinematics_transform(
        pose_skeleton, global_orient, phis,
        rest_J.clone(), children, parents, dtype=dtype, train=train,
        leaf_thetas=leaf_thetas)

    test_joints = True
    if test_joints:
        J_transformed, A = batch_rigid_transform(rot_mats, rest_J[:, :24].clone(), parents[:24], dtype=dtype)
    else:
        J_transformed = None

    # assert torch.mean(torch.abs(rotate_rest_pose - J_transformed)) < 1e-5
    # 4. Add pose blend shapes
    # rot_mats: N x (J + 1) x 3 x 3
    ident = torch.eye(3, dtype=dtype, device=device)
    pose_feature = (rot_mats[:, 1:] - ident).view([batch_size, -1])
    pose_offsets = torch.matmul(pose_feature, posedirs) \
        .view(batch_size, -1, 3)

    v_posed = pose_offsets + v_shaped

    # 5. Do skinning:
    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                               dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]
    J_from_verts_h36m = vertices2joints(J_regressor_h36m, verts)

    return verts, J_transformed, rot_mats, J_from_verts_h36m

def marker_ik(betas, global_orient, pose_skeleton, obs_marker,
           v_template, shapedirs, posedirs, J_regressor, J_regressor_h36m, parents, children,
           lbs_weights, dtype=torch.float32, train=False, leaf_thetas=None, T_markers=None):
    ''' Performs Linear Blend Skinning with the given shape and skeleton joints

        Parameters
        ----------
        betas : torch.tensor BxNB
            The tensor of shape parameters
        global_orient : torch.tensor Bx3
            The tensor of global orientation
        pose_skeleton : torch.tensor BxJ*3
            The pose skeleton in (X, Y, Z) format
        phis : torch.tensor BxJx2
            The rotation on bone axis parameters
        v_template torch.tensor BxVx3
            The template mesh that will be deformed
        shapedirs : torch.tensor 1xNB
            The tensor of PCA shape displacements
        posedirs : torch.tensor Px(V * 3)
            The pose PCA coefficients
        J_regressor : torch.tensor JxV
            The regressor array that is used to calculate the joints from
            the position of the vertices
        J_regressor_h36m : torch.tensor 17xV
            The regressor array that is used to calculate the 17 Human3.6M joints from
            the position of the vertices
        parents: torch.tensor J
            The array that describes the kinematic parents for the model
        children: dict
            The dictionary that describes the kinematic chidrens for the model
        lbs_weights: torch.tensor N x V x (J + 1)
            The linear blend skinning weights that represent how much the
            rotation matrix of each part affects each vertex
        dtype: torch.dtype, optional

        Returns
        -------
        verts: torch.tensor BxVx3
            The vertices of the mesh after applying the shape and pose
            displacements.
        joints: torch.tensor BxJx3
            The joints of the model
        rot_mats: torch.tensor BxJx3x3
            The rotation matrics of each joints
    '''
    batch_size = max(betas.shape[0], pose_skeleton.shape[0])
    device = betas.device

    # 1. Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    # 2. Get the rest joints
    # NxJx3 array
    rest_J = vertices2joints(J_regressor, v_shaped)
    rest_markers = T_markers + rest_J[:, :1]
    # if leaf_thetas is not None:
    #     rest_J = vertices2joints(J_regressor, v_shaped)
    # else:
    #     rest_J = torch.zeros((v_shaped.shape[0], 29, 3), dtype=dtype, device=device)
    #     rest_J[:, :24] = vertices2joints(J_regressor, v_shaped)
    #
    #     leaf_number = [411, 2445, 5905, 3216, 6617]
    #     leaf_vertices = v_shaped[:, leaf_number].clone()
    #     rest_J[:, 24:] = leaf_vertices

    # 3. Get the rotation matrics
    rot_mats, rotate_rest_pose = batch_inverse_kinematics_transform_marker(
        pose_skeleton, global_orient, obs_marker, rest_markers.clone(),
        rest_J.clone(), children, parents, dtype=dtype, train=train,
        leaf_thetas=leaf_thetas)

    test_joints = True
    if test_joints:
        J_transformed, A = batch_rigid_transform(rot_mats, rest_J[:, :24].clone(), parents[:24], dtype=dtype)
    else:
        J_transformed = None

    # assert torch.mean(torch.abs(rotate_rest_pose - J_transformed)) < 1e-5
    # 4. Add pose blend shapes
    # rot_mats: N x (J + 1) x 3 x 3
    ident = torch.eye(3, dtype=dtype, device=device)
    pose_feature = (rot_mats[:, 1:] - ident).view([batch_size, -1])
    pose_offsets = torch.matmul(pose_feature, posedirs) \
        .view(batch_size, -1, 3)

    v_posed = pose_offsets + v_shaped

    # 5. Do skinning:
    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                               dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]
    J_from_verts_h36m = vertices2joints(J_regressor_h36m, verts)

    return verts, J_transformed, rot_mats, J_from_verts_h36m

def vertices2joints(J_regressor, vertices):
    ''' Calculates the 3D joint locations from the vertices

    Parameters
    ----------
    J_regressor : torch.tensor JxV
        The regressor array that is used to calculate the joints from the
        position of the vertices
    vertices : torch.tensor BxVx3
        The tensor of mesh vertices

    Returns
    -------
    torch.tensor BxJx3
        The location of the joints
    '''

    return torch.einsum('bik,ji->bjk', [vertices, J_regressor])


def blend_shapes(betas, shape_disps):
    ''' Calculates the per vertex displacement due to the blend shapes


    Parameters
    ----------
    betas : torch.tensor Bx(num_betas)
        Blend shape coefficients
    shape_disps: torch.tensor Vx3x(num_betas)
        Blend shapes

    Returns
    -------
    torch.tensor BxVx3
        The per-vertex displacement due to shape deformation
    '''

    # Displacement[b, m, k] = sum_{l} betas[b, l] * shape_disps[m, k, l]
    # i.e. Multiply each shape displacement by its corresponding beta and
    # then sum them.
    blend_shape = torch.einsum('bl,mkl->bmk', [betas, shape_disps])
    return blend_shape


def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def transform_mat(R, t):
    ''' Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]),
                      F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def batch_rigid_transform(rot_mats, joints, parents, dtype=torch.float32):
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints. (Template Pose)
    parents : torch.tensor BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32

    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """
    joints = torch.unsqueeze(joints, dim=-1)
    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]].clone()

    # (B, K + 1, 4, 4)
    transforms_mat = transform_mat(
        rot_mats.reshape(-1, 3, 3),
        rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        # (B, 4, 4) x (B, 4, 4)
        curr_res = torch.matmul(transform_chain[parents[i]],
                                transforms_mat[:, i])
        transform_chain.append(curr_res)

    # (B, K + 1, 4, 4)
    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = F.pad(joints, [0, 0, 0, 1])

    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

    return posed_joints, rel_transforms


def batch_inverse_kinematics_transform(
        pose_skeleton, global_orient,
        phis,
        rest_pose,
        children, parents, dtype=torch.float32, train=False,
        leaf_thetas=None):
    """
    Applies a batch of inverse kinematics transfoirm to the joints

    Parameters
    ----------
    pose_skeleton : torch.tensor BxNx3
        Locations of estimated pose skeleton.
    global_orient : torch.tensor Bx1x3x3
        Tensor of global rotation matrices
    phis : torch.tensor BxNx2
        The rotation on bone axis parameters
    rest_pose : torch.tensor Bx(N+1)x3
        Locations of rest_pose. (Template Pose)
    children: dict
        The dictionary that describes the kinematic chidrens for the model
    parents : torch.tensor Bx(N+1)
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32

    Returns
    -------
    rot_mats: torch.tensor Bx(N+1)x3x3
        The rotation matrics of each joints
    rel_transforms : torch.tensor Bx(N+1)x4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """
    batch_size = pose_skeleton.shape[0]
    device = pose_skeleton.device

    rel_rest_pose = rest_pose.clone()
    rel_rest_pose[:, 1:] -= rest_pose[:, parents[1:]].clone()
    rel_rest_pose = torch.unsqueeze(rel_rest_pose, dim=-1)

    # rotate the T pose
    rotate_rest_pose = torch.zeros_like(rel_rest_pose)
    # set up the root
    rotate_rest_pose[:, 0] = rel_rest_pose[:, 0]

    rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1).detach()
    rel_pose_skeleton[:, 1:] = rel_pose_skeleton[:, 1:] - rel_pose_skeleton[:, parents[1:]].clone()
    rel_pose_skeleton[:, 0] = rel_rest_pose[:, 0]

    # the predicted final pose
    final_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1)
    final_pose_skeleton = final_pose_skeleton - final_pose_skeleton[:, 0:1] + rel_rest_pose[:, 0:1]

    assert phis.dim() == 3
    phis = phis / (torch.norm(phis, dim=2, keepdim=True) + 1e-8)

    # TODO
    if train:
        global_orient_mat = batch_get_pelvis_orient(
            rel_pose_skeleton.clone(), rel_rest_pose.clone(), parents, children, dtype)
    else:
        global_orient_mat = batch_get_pelvis_orient_svd(
            rel_pose_skeleton.clone(), rel_rest_pose.clone(), parents, children, dtype)

    rot_mat_chain = [global_orient_mat]
    rot_mat_local = [global_orient_mat]
    # leaf nodes rot_mats
    if leaf_thetas is not None:
        leaf_cnt = 0
        leaf_rot_mats = leaf_thetas.view([batch_size, 5, 3, 3])

    for i in range(1, parents.shape[0]):
        if children[i] == -1:
            # leaf nodes
            if leaf_thetas is not None:
                rot_mat = leaf_rot_mats[:, leaf_cnt, :, :]
                leaf_cnt += 1

                rotate_rest_pose[:, i] = rotate_rest_pose[:, parents[i]] + torch.matmul(
                    rot_mat_chain[parents[i]],
                    rel_rest_pose[:, i]
                )

                rot_mat_chain.append(torch.matmul(
                    rot_mat_chain[parents[i]],
                    rot_mat))
                rot_mat_local.append(rot_mat)
        elif children[i] == -3:
            # three children
            rotate_rest_pose[:, i] = rotate_rest_pose[:, parents[i]] + torch.matmul(
                rot_mat_chain[parents[i]],
                rel_rest_pose[:, i]
            )

            spine_child = []
            for c in range(1, parents.shape[0]):
                if parents[c] == i and c not in spine_child:
                    spine_child.append(c)

            # original
            spine_child = []
            for c in range(1, parents.shape[0]):
                if parents[c] == i and c not in spine_child:
                    spine_child.append(c)

            children_final_loc = []
            children_rest_loc = []
            for c in spine_child:
                temp = final_pose_skeleton[:, c] - rotate_rest_pose[:, i]
                children_final_loc.append(temp)

                children_rest_loc.append(rel_rest_pose[:, c].clone())

            rot_mat = batch_get_3children_orient_svd(
                children_final_loc, children_rest_loc,
                rot_mat_chain[parents[i]], spine_child, dtype)

            rot_mat_chain.append(
                torch.matmul(
                    rot_mat_chain[parents[i]],
                    rot_mat)
            )
            rot_mat_local.append(rot_mat)
        else:
            rotate_rest_pose[:, i] = rotate_rest_pose[:, parents[i]] + torch.matmul(
                rot_mat_chain[parents[i]],
                rel_rest_pose[:, i]
            )
            # (B, 3, 1)
            child_final_loc = final_pose_skeleton[:, children[i]] - rotate_rest_pose[:, i]

            if not train:
                orig_vec = rel_pose_skeleton[:, children[i]]
                template_vec = rel_rest_pose[:, children[i]]
                norm_t = torch.norm(template_vec, dim=1, keepdim=True)
                orig_vec = orig_vec * norm_t / torch.norm(orig_vec, dim=1, keepdim=True)

                diff = torch.norm(child_final_loc - orig_vec, dim=1, keepdim=True)
                big_diff_idx = torch.where(diff > 15 / 1000)[0]

                child_final_loc[big_diff_idx] = orig_vec[big_diff_idx]

            child_final_loc = torch.matmul(
                rot_mat_chain[parents[i]].transpose(1, 2),
                child_final_loc)

            child_rest_loc = rel_rest_pose[:, children[i]]
            # (B, 1, 1)
            child_final_norm = torch.norm(child_final_loc, dim=1, keepdim=True)
            child_rest_norm = torch.norm(child_rest_loc, dim=1, keepdim=True)

            child_final_norm = torch.norm(child_final_loc, dim=1, keepdim=True)
            # 旋转前后的向量确定swing的转轴和转角
            # (B, 3, 1)
            axis = torch.cross(child_rest_loc, child_final_loc, dim=1)
            axis_norm = torch.norm(axis, dim=1, keepdim=True)

            # (B, 1, 1)
            cos = torch.sum(child_rest_loc * child_final_loc, dim=1, keepdim=True) / (child_rest_norm * child_final_norm + 1e-8)
            sin = axis_norm / (child_rest_norm * child_final_norm + 1e-8)

            # (B, 3, 1)
            axis = axis / (axis_norm + 1e-8)

            # Convert location revolve to rot_mat by rodrigues
            # (B, 1, 1)
            rx, ry, rz = torch.split(axis, 1, dim=1)
            zeros = torch.zeros((batch_size, 1, 1), dtype=dtype, device=device)

            K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
                .view((batch_size, 3, 3))
            ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
            # swing 的rotmat
            rot_mat_loc = ident + sin * K + (1 - cos) * torch.bmm(K, K)

            # Convert spin to rot_mat
            # (B, 3, 1)
            # twist的转轴直接用Tpose里的
            spin_axis = child_rest_loc / child_rest_norm
            # (B, 1, 1)
            rx, ry, rz = torch.split(spin_axis, 1, dim=1)
            zeros = torch.zeros((batch_size, 1, 1), dtype=dtype, device=device)
            K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
                .view((batch_size, 3, 3))
            ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
            # (B, 1, 1)
            cos, sin = torch.split(phis[:, i - 1], 1, dim=1)
            cos = torch.unsqueeze(cos, dim=2)
            sin = torch.unsqueeze(sin, dim=2)
            rot_mat_spin = ident + sin * K + (1 - cos) * torch.bmm(K, K)
            # rot_mat_spin = ident
            rot_mat = torch.matmul(rot_mat_loc, rot_mat_spin)

            rot_mat_chain.append(torch.matmul(
                rot_mat_chain[parents[i]],
                rot_mat))
            rot_mat_local.append(rot_mat)

    # (B, K + 1, 3, 3)
    rot_mats = torch.stack(rot_mat_local, dim=1)

    return rot_mats, rotate_rest_pose.squeeze(-1)

def batch_inverse_kinematics_transform_marker(
        pose_skeleton, global_orient,
        markers,
        rest_markers,
        rest_pose,
        children, parents, dtype=torch.float32, train=False,
        leaf_thetas=None):
    marker2joint = [1,  2,  0,  0, 18, 19, 16, 17, 10, 11,  1,  2,  9,  9, 16, 17, 20, 21,
        20, 21,  4,  5,  1,  2,  7,  8, 13, 14,  6,  6, 20, 21]
    # marker2joint = [0, 0, 0, 0, 18, 19, 16, 17, 10, 11, 0, 0, 6, 6, 16, 17, 22, 23, 20, 21, 4, 5, 1, 2, 7, 8, 0, 0, 3, 3, 22, 23]
    
    joint2marker = [[],[22],[23],[28],[20],[21],[12],[24],[25],[],[8],[9],[],[],[],[],[14],[15],[6],[7],[30],[31],[30],[31]]
    # joint2marker = [[],[22],[23],[28, 29],[20],[21],[12, 13],[24],[25],[],[8],[9],[],[],[],[],[6, 14],[7,15],[18],[19],[18],[19],[16, 30],[17, 31]]
    batch_size = pose_skeleton.shape[0]
    device = pose_skeleton.device

    rel_rest_pose = rest_pose.clone() # (N, 24, 3)
    rel_rest_pose[:, 1:] -= rest_pose[:, parents[1:]].clone()

    rel_rest_marker = rest_markers.clone() # (N, 32, 3)
    rel_rest_marker -= rest_pose[:, marker2joint].clone()
    #print(rest_markers[0, 22] - rest_pose[0, 1])
    #exit()

    rel_rest_pose = torch.unsqueeze(rel_rest_pose, dim=-1)
    rel_rest_marker = torch.unsqueeze(rel_rest_marker, dim=-1)
   
    # rotate the T pose
    rotate_rest_pose = torch.zeros_like(rel_rest_pose)
    # set up the root
    rotate_rest_pose[:, 0] = rel_rest_pose[:, 0]

    # for global pose estimation
    rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1).detach()
    rel_pose_skeleton[:, 1:] = rel_pose_skeleton[:, 1:] - rel_pose_skeleton[:, parents[1:]].clone()
    rel_pose_skeleton[:, 0] = rel_rest_pose[:, 0]

    # the predicted final pose
    final_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1)
    final_pose_skeleton = final_pose_skeleton - final_pose_skeleton[:, 0:1] + rel_rest_pose[:, 0:1]

    final_markers_skeleton = torch.unsqueeze(markers.clone(), dim=-1)
    final_markers_skeleton = final_markers_skeleton - torch.unsqueeze(pose_skeleton.clone(), dim=-1)[:, 0:1] + rel_rest_pose[:, 0:1]
    # print(final_markers_skeleton.shape)
    # from vis_tools import Visualizer
    # vis = Visualizer()
    # vis.show_points([final_markers_skeleton.reshape(batch_size, -1, 3),
    #                 #final_pose_skeleton.reshape(batch_size, -1, 3),
    #                 ], fix_axis=True)

    #final_markers_skeleton (n, 32, 3, 1)

    #assert phis.dim() == 3
    #phis = phis / (torch.norm(phis, dim=2, keepdim=True) + 1e-8)

    # TODO
    if train:
        global_orient_mat = batch_get_pelvis_orient(
            rel_pose_skeleton.clone(), rel_rest_pose.clone(), parents, children, dtype)
    else:
        global_orient_mat = batch_get_pelvis_orient_svd(
            rel_pose_skeleton.clone(), rel_rest_pose.clone(), parents, children, dtype)

    rot_mat_chain = [global_orient_mat]
    rot_mat_local = [global_orient_mat]
    #print(children)
    # leaf nodes rot_mats
    for i in range(1, parents.shape[0]):
        # 没有child节点 -> 考虑是否有marker
        if children[i] <= 0:
            ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0).repeat(batch_size, 1, 1)
            rot_mat_loc = ident
            if len(joint2marker[i]) == 0:
                rot_mat_spin = ident
            else:
                # rot_mat_spin = ident
                rotate_rest_pose[:, i] = rotate_rest_pose[:, parents[i]] + torch.matmul(
                    rot_mat_chain[parents[i]],
                    rel_rest_pose[:, i]
                )
                marker_final_loc = final_markers_skeleton[:, joint2marker[i]] - rotate_rest_pose[:, i:i+1]
                marker_final_loc = torch.matmul(
                    rot_mat_chain[parents[i]].transpose(1, 2)[:, None], # (B, 3, 3)
                    marker_final_loc # (B, N, 3, 1)
                ) # (B, N, 3, 1)
                # (B, 3, 1)
                marker_rest_loc = rel_rest_marker[:, joint2marker[i]]  # (B, N, 3, 1)
                #rot_mat_spin = batch_get_rot_mat_svd(marker_final_loc, marker_rest_loc)

                rot_mat_spin = vectors2rotmat(marker_rest_loc.mean(1), marker_final_loc.mean(1), dtype)

            rot_mat = torch.matmul(rot_mat_loc, rot_mat_spin)
            #leaf_cnt += 1

            rotate_rest_pose[:, i] = rotate_rest_pose[:, parents[i]] + torch.matmul(
                rot_mat_chain[parents[i]],
                rel_rest_pose[:, i]
            )

            rot_mat_chain.append(torch.matmul(
                rot_mat_chain[parents[i]],
                rot_mat))
            rot_mat_local.append(rot_mat)
        else:
            rotate_rest_pose[:, i] = rotate_rest_pose[:, parents[i]] + torch.matmul(
                rot_mat_chain[parents[i]],
                rel_rest_pose[:, i]
            )
            # (B, 3, 1)
            child_final_loc = final_pose_skeleton[:, children[i]] - rotate_rest_pose[:, i]

            # if not train:
            #     orig_vec = rel_pose_skeleton[:, children[i]]
            #     template_vec = rel_rest_pose[:, children[i]]
            #     norm_t = torch.norm(template_vec, dim=1, keepdim=True)
            #     orig_vec = orig_vec * norm_t / torch.norm(orig_vec, dim=1, keepdim=True)

            #     diff = torch.norm(child_final_loc - orig_vec, dim=1, keepdim=True)
            #     big_diff_idx = torch.where(diff > 15 / 1000)[0]

            #     child_final_loc[big_diff_idx] = orig_vec[big_diff_idx]

            child_final_loc = torch.matmul(
                rot_mat_chain[parents[i]].transpose(1, 2),
                child_final_loc)
            child_rest_loc = rel_rest_pose[:, children[i]]
            # (B, 1, 1)
            child_final_norm = torch.norm(child_final_loc, dim=1, keepdim=True)
            child_rest_norm = torch.norm(child_rest_loc, dim=1, keepdim=True)

            # 旋转前后的向量确定swing的转轴和转角
            # (B, 3, 1)
            axis = torch.cross(child_rest_loc, child_final_loc, dim=1)
            axis_norm = torch.norm(axis, dim=1, keepdim=True)

            # (B, 1, 1)
            cos = torch.sum(child_rest_loc * child_final_loc, dim=1, keepdim=True) / (child_rest_norm * child_final_norm + 1e-8)
            sin = axis_norm / (child_rest_norm * child_final_norm + 1e-8)

            # (B, 3, 1)
            axis = axis / (axis_norm + 1e-8)

            # Convert location revolve to rot_mat by rodrigues
            # (B, 1, 1)
            rx, ry, rz = torch.split(axis, 1, dim=1)
            zeros = torch.zeros((batch_size, 1, 1), dtype=dtype, device=device)

            K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
                .view((batch_size, 3, 3))
            ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
            # swing 的rotmat
            rot_mat_loc = ident + sin * K + (1 - cos) * torch.bmm(K, K)

            # Convert spin to rot_mat
            # (B, 3, 1)
            # twist的转轴直接用Tpose里的
            spin_axis = child_rest_loc / child_rest_norm

            # (B, 3, 1)
            # obs_skeleton - 旋转后的根节点的位置
            if len(joint2marker[i]) == 0:
                rot_mat_spin = ident
            else:
                rot_mat_spin = ident
                marker_final_loc = final_markers_skeleton[:, joint2marker[i]].mean(1) - rotate_rest_pose[:, i]
                marker_final_loc = torch.matmul(
                    rot_mat_chain[parents[i]].transpose(1, 2),
                    marker_final_loc
                )
                marker_rest_loc = rel_rest_marker[:, joint2marker[i]].mean(1)
                
                marker_final_norm = torch.norm(marker_final_loc, dim=1, keepdim=True)
                marker_rest_norm = torch.norm(marker_rest_loc, dim=1, keepdim=True)

                marker_final_loc = marker_final_loc / (marker_final_norm + 1e-8)
                marker_rest_loc = marker_rest_loc / (marker_rest_norm + 1e-8)
                

                # Project vectors onto plane perpendicular to spin axis
                a1 = (marker_final_loc * spin_axis).sum(1, keepdim=True) * spin_axis
                a2 = marker_final_loc - a1
                a2_norm = torch.norm(a2, dim=1, keepdim=True)
                # mask = a2_norm > 1e-6
                # a2 = torch.where(mask, a2 / (a2_norm + 1e-8), torch.zeros_like(a2))

                b1 = (marker_rest_loc * spin_axis).sum(1, keepdim=True) * spin_axis
                b2 = marker_rest_loc - b1
                b2_norm = torch.norm(b2, dim=1, keepdim=True)
                # mask = b2_norm > 1e-6
                # b2 = torch.where(mask, b2 / (b2_norm + 1e-8), torch.zeros_like(b2))

                _dir = torch.cross(a2, b2, dim=1)
                _dir_rev = torch.cross(b2, a2, dim=1)
                mask = (torch.sum(_dir * spin_axis, dim=1) > 0).flatten()
                _dir = torch.where(mask[:, None, None], _dir, _dir_rev)
                #print(mask)

                cos = torch.sum(a2 * b2, dim=1, keepdim=True) / (a2_norm * b2_norm + 1e-8)
                sin = torch.norm(_dir, dim=1, keepdim=True) / (a2_norm * b2_norm + 1e-8)
                theta = torch.atan2(sin, cos) # (B, 1, 1)
                theta = torch.where(theta > torch.pi/2, theta * 0, theta )
                print(i, theta * 180. / torch.pi)
                #mask = theta > 

                rotvec = spin_axis * theta
              
                rot_mat_spin = roma.rotvec_to_rotmat(rotvec.reshape(-1, 3))


            rot_mat = torch.matmul(rot_mat_loc, rot_mat_spin)

            # # rot_mat_spin = ident

            rot_mat_chain.append(torch.matmul(
                rot_mat_chain[parents[i]],
                rot_mat))
            rot_mat_local.append(rot_mat)

    # (B, K + 1, 3, 3)
    rot_mats = torch.stack(rot_mat_local, dim=1)

    return rot_mats, rotate_rest_pose.squeeze(-1)

def batch_get_pelvis_orient_svd(rel_pose_skeleton, rel_rest_pose, parents, children, dtype):
    pelvis_child = [int(children[0])]
    for i in range(1, parents.shape[0]):
        if parents[i] == 0 and i not in pelvis_child:
            pelvis_child.append(i)

    rest_mat = []
    target_mat = []
    for child in pelvis_child:
        rest_mat.append(rel_rest_pose[:, child].clone())
        target_mat.append(rel_pose_skeleton[:, child].clone())

    rest_mat = torch.cat(rest_mat, dim=2)
    target_mat = torch.cat(target_mat, dim=2)
    S = rest_mat.bmm(target_mat.transpose(1, 2))

    mask_zero = S.sum(dim=(1, 2))

    S_non_zero = S[mask_zero != 0].reshape(-1, 3, 3)

    U, _, V = torch.svd(S_non_zero)

    rot_mat = torch.zeros_like(S)
    rot_mat[mask_zero == 0] = torch.eye(3, device=S.device)

    rot_mat_non_zero = torch.bmm(V, U.transpose(1, 2))
    rot_mat[mask_zero != 0] = rot_mat_non_zero

    assert torch.sum(torch.isnan(rot_mat)) == 0, ('rot_mat', rot_mat)

    return rot_mat


def batch_get_pelvis_orient(rel_pose_skeleton, rel_rest_pose, parents, children, dtype):
    batch_size = rel_pose_skeleton.shape[0]
    device = rel_pose_skeleton.device

    assert children[0] == 3
    pelvis_child = [int(children[0])]
    for i in range(1, parents.shape[0]):
        if parents[i] == 0 and i not in pelvis_child:
            pelvis_child.append(i)

    spine_final_loc = rel_pose_skeleton[:, int(children[0])].clone()
    spine_rest_loc = rel_rest_pose[:, int(children[0])].clone()
    spine_norm = torch.norm(spine_final_loc, dim=1, keepdim=True)
    spine_norm = spine_final_loc / (spine_norm + 1e-8)

    rot_mat_spine = vectors2rotmat(spine_rest_loc, spine_final_loc, dtype)

    assert torch.sum(torch.isnan(rot_mat_spine)
                     ) == 0, ('rot_mat_spine', rot_mat_spine)
    center_final_loc = 0
    center_rest_loc = 0
    for child in pelvis_child:
        if child == int(children[0]):
            continue
        center_final_loc = center_final_loc + rel_pose_skeleton[:, child].clone()
        center_rest_loc = center_rest_loc + rel_rest_pose[:, child].clone()
    center_final_loc = center_final_loc / (len(pelvis_child) - 1)
    center_rest_loc = center_rest_loc / (len(pelvis_child) - 1)

    center_rest_loc = torch.matmul(rot_mat_spine, center_rest_loc)

    center_final_loc = center_final_loc - torch.sum(center_final_loc * spine_norm, dim=1, keepdim=True) * spine_norm
    center_rest_loc = center_rest_loc - torch.sum(center_rest_loc * spine_norm, dim=1, keepdim=True) * spine_norm

    center_final_loc_norm = torch.norm(center_final_loc, dim=1, keepdim=True)
    center_rest_loc_norm = torch.norm(center_rest_loc, dim=1, keepdim=True)

    # (B, 3, 1)
    axis = torch.cross(center_rest_loc, center_final_loc, dim=1)
    axis_norm = torch.norm(axis, dim=1, keepdim=True)

    # (B, 1, 1)
    cos = torch.sum(center_rest_loc * center_final_loc, dim=1, keepdim=True) / (center_rest_loc_norm * center_final_loc_norm + 1e-8)
    sin = axis_norm / (center_rest_loc_norm * center_final_loc_norm + 1e-8)

    assert torch.sum(torch.isnan(cos)
                     ) == 0, ('cos', cos)
    assert torch.sum(torch.isnan(sin)
                     ) == 0, ('sin', sin)
    # (B, 3, 1)
    axis = axis / (axis_norm + 1e-8)

    # Convert location revolve to rot_mat by rodrigues
    # (B, 1, 1)
    rx, ry, rz = torch.split(axis, 1, dim=1)
    zeros = torch.zeros((batch_size, 1, 1), dtype=dtype, device=device)

    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))
    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat_center = ident + sin * K + (1 - cos) * torch.bmm(K, K)

    rot_mat = torch.matmul(rot_mat_center, rot_mat_spine)

    return rot_mat


def batch_get_3children_orient_svd(rel_pose_skeleton, rel_rest_pose, rot_mat_chain_parent, children_list, dtype):
    rest_mat = []
    target_mat = []
    for c, child in enumerate(children_list):
        if isinstance(rel_pose_skeleton, list):
            target = rel_pose_skeleton[c].clone()
            template = rel_rest_pose[c].clone()
        else:
            target = rel_pose_skeleton[:, child].clone()
            template = rel_rest_pose[:, child].clone()

        target = torch.matmul(
            rot_mat_chain_parent.transpose(1, 2),
            target)

        target_mat.append(target)
        rest_mat.append(template)

    rest_mat = torch.cat(rest_mat, dim=2)
    target_mat = torch.cat(target_mat, dim=2)
    S = rest_mat.bmm(target_mat.transpose(1, 2))

    U, _, V = torch.svd(S)

    rot_mat = torch.bmm(V, U.transpose(1, 2))
    assert torch.sum(torch.isnan(rot_mat)) == 0, ('3children rot_mat', rot_mat)
    return rot_mat


def batch_get_rot_mat_svd(rel_pose_skeleton, rel_rest_pose):
    # rel_pose_skeleton B, N, 3, 1
    # rel_rest_pose B, N, 3, 1
    print(rel_pose_skeleton.shape, rel_rest_pose.shape)
    S = rel_rest_pose.transpose(1, 2)[..., 0].bmm(rel_pose_skeleton[..., 0])

    U, _, V = torch.svd(S)

    rot_mat = torch.bmm(V, U.transpose(1, 2)).transpose(-1, -2)
    assert torch.sum(torch.isnan(rot_mat)) == 0, ('3children rot_mat', rot_mat)
    return rot_mat


def vectors2rotmat(vec_rest, vec_final, dtype):
    batch_size = vec_final.shape[0]
    device = vec_final.device

    # (B, 1, 1)
    vec_final_norm = torch.norm(vec_final, dim=1, keepdim=True)
    vec_rest_norm = torch.norm(vec_rest, dim=1, keepdim=True)

    # (B, 3, 1)
    axis = torch.cross(vec_rest, vec_final, dim=1)
    axis_norm = torch.norm(axis, dim=1, keepdim=True)

    # (B, 1, 1)
    cos = torch.sum(vec_rest * vec_final, dim=1, keepdim=True) / (vec_rest_norm * vec_final_norm + 1e-8)
    sin = axis_norm / (vec_rest_norm * vec_final_norm + 1e-8)

    # (B, 3, 1)
    axis = axis / (axis_norm + 1e-8)

    # Convert location revolve to rot_mat by rodrigues
    # (B, 1, 1)
    rx, ry, rz = torch.split(axis, 1, dim=1)
    zeros = torch.zeros((batch_size, 1, 1), dtype=dtype, device=device)

    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))
    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat_loc = ident + sin * K + (1 - cos) * torch.bmm(K, K)

    return rot_mat_loc


def rotmat_to_quat(rotmat):
    """Convert rotation matrix to quaternion coefficients.
    Args:
        rotmat: size is [B, 3, 3]
    Returns:
        Quaternion: size is [B, 4] <===> (w, x, y, z)
    """
    quaternion = torch.zeros([rotmat.size(0), 4], device=rotmat.device)
    trace = rotmat[:, 0, 0] + rotmat[:, 1, 1] + rotmat[:, 2, 2]
    flag = 1 + trace > 0
    s = torch.zeros_like(trace)

    # pos
    s[flag] = 2 * torch.sqrt(1 + trace[flag]) + 1e-16
    s_pos = s[flag]
    quaternion[flag, 0] = s_pos / 4
    quaternion[flag, 1] = (rotmat[flag, 2, 1] - rotmat[flag, 1, 2]) / s_pos
    quaternion[flag, 2] = (rotmat[flag, 0, 2] - rotmat[flag, 2, 0]) / s_pos
    quaternion[flag, 3] = (rotmat[flag, 1, 0] - rotmat[flag, 0, 1]) / s_pos

    # neg
    diag = torch.stack([rotmat[:, 0, 0], rotmat[:, 1, 1], rotmat[:, 2, 2]])
    max_val, max_ind = torch.max(diag, dim=0)

    s[~flag] = 2 * torch.sqrt(1 - trace[~flag] + 2 * max_val[~flag]) + 1e-16

    f0 = ~flag * (max_ind == 0)
    s0 = s[f0]
    quaternion[f0, 0] = (rotmat[f0, 2, 1] - rotmat[f0, 1, 2]) / s0
    quaternion[f0, 1] = s0 / 4
    quaternion[f0, 2] = (rotmat[f0, 0, 1] + rotmat[f0, 1, 0]) / s0
    quaternion[f0, 3] = (rotmat[f0, 0, 2] + rotmat[f0, 2, 0]) / s0

    f1 = ~flag * (max_ind == 1)
    s1 = s[f1]
    quaternion[f1, 0] = (rotmat[f1, 0, 2] - rotmat[f1, 2, 0]) / s1
    quaternion[f1, 1] = (rotmat[f1, 0, 1] + rotmat[f1, 1, 0]) / s1
    quaternion[f1, 2] = s1 / 4
    quaternion[f1, 3] = (rotmat[f1, 1, 2] + rotmat[f1, 2, 1]) / s1

    f2 = ~flag * (max_ind == 2)
    s2 = s[f2]
    quaternion[f2, 0] = (rotmat[f2, 1, 0] - rotmat[f2, 0, 1]) / s2
    quaternion[f2, 1] = (rotmat[f2, 0, 2] + rotmat[f2, 2, 0]) / s2
    quaternion[f2, 2] = (rotmat[f2, 1, 2] + rotmat[f2, 2, 1]) / s2
    quaternion[f2, 3] = s2 / 4

    return quaternion


def quat_to_rotmat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / (norm_quat.norm(p=2, dim=1, keepdim=True) + 1e-8)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                          2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                          2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat
def get_phi(rot_mats, joints, parents):
    joints = torch.unsqueeze(joints, dim=-1)
    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]].clone()

    # modified by xuchao
    childs = -torch.ones((parents.shape[0]), dtype=parents.dtype, device=parents.device)
    for i in range(1, parents.shape[0]):
        childs[parents[i]] = i

    dtype = rot_mats.dtype
    batch_size = rot_mats.shape[0]
    device = rot_mats.device

    angle_twist = []
    # error = False
    for i in range(1, rot_mats.shape[1]):
        # modified by xuchao
        if childs[i] < 0:
            angle_twist.append(torch.zeros((batch_size, 1), dtype=rot_mats.dtype, device=rot_mats.device))
            continue

        u = rel_joints[:, childs[i]]
        rot = rot_mats[:, i]

        v = torch.matmul(rot, u)

        u_norm = torch.norm(u, dim=1, keepdim=True)
        v_norm = torch.norm(v, dim=1, keepdim=True)

        axis = torch.cross(u, v, dim=1) # swing axis
        axis_norm = torch.norm(axis, dim=1, keepdim=True)

        # (B, 1, 1)
        cos = torch.sum(u * v, dim=1, keepdim=True) / (u_norm * v_norm + 1e-8)
        sin = axis_norm / (u_norm * v_norm + 1e-8)

        # (B, 3, 1)
        axis = axis / (axis_norm + 1e-8)

        # Convert location revolve to rot_mat by rodrigues
        # (B, 1, 1)
        rx, ry, rz = torch.split(axis, 1, dim=1)
        zeros = torch.zeros((batch_size, 1, 1), dtype=dtype, device=device)

        K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
            .view((batch_size, 3, 3))
        ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
        rot_mat_pivot = ident + sin * K + (1 - cos) * torch.bmm(K, K)

        rot_mat_twist = torch.matmul(rot_mat_pivot.transpose(1, 2), rot)

        rot_aa_twist = roma.rotmat_to_rotvec(rot_mat_twist)
        angle = torch.norm(rot_aa_twist, dim=1, keepdim=True)
        axis = rot_aa_twist / angle

        axis = axis / torch.norm(axis, dim=1, keepdim=True)
        spin_axis = u / u_norm
        spin_axis = spin_axis.squeeze(-1)

        pos = torch.norm(spin_axis - axis, dim=1)
        neg = torch.norm(spin_axis + axis, dim=1)
        #
        _angle = torch.zeros_like(angle)# * -999
        mask = neg < pos
        _angle[mask & (pos > 1.8)] = -angle[mask&(pos>1.8)]
        _angle[(~mask) & (neg > 1.8)] = angle[(~mask)&(neg>1.8)]
        angle_twist.append(_angle)


    angle_twist = torch.cat(angle_twist, dim=1) # (b, 23)
    cos = torch.cos(angle_twist)
    sin = torch.sin(angle_twist)
    return torch.stack([cos, sin], dim =2)


def batch_inverse_kinematics_decompose(
        pose_skeleton,
        phis,
        rest_pose,
        children, parents, dtype=torch.float32,
        leaf_thetas=None, need_detach=True):
    """
    Applies a batch of inverse kinematics transfoirm to the joints

    Parameters
    ----------
    pose_skeleton : torch.tensor BxNx3
        Locations of estimated pose skeleton.
    global_orient : torch.tensor Bx1x3x3
        Tensor of global rotation matrices
    phis : torch.tensor BxNx2
        The rotation on bone axis parameters
    rest_pose : torch.tensor Bx(N+1)x3
        Locations of rest_pose. (Template Pose)
    children: dict
        The dictionary that describes the kinematic chidrens for the model
    parents : torch.tensor Bx(N+1)
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32

    Returns
    -------
    rot_mats: torch.tensor Bx(N+1)x3x3
        The rotation matrics of each joints
    rel_transforms : torch.tensor Bx(N+1)x4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """
    batch_size = pose_skeleton.shape[0]
    device = pose_skeleton.device

    rel_rest_pose = rest_pose.clone()
    rel_rest_pose[:, 1:] -= rest_pose[:, parents[1:]].clone()
    rel_rest_pose = torch.unsqueeze(rel_rest_pose, dim=-1)
    unit_rel_rest_pose = rel_rest_pose / torch.norm(rel_rest_pose, dim=2, keepdim=True)

    # rotate the T pose
    rotate_rest_pose = torch.zeros_like(rel_rest_pose)
    # set up the root
    rotate_rest_pose[:, 0] = rel_rest_pose[:, 0]

    if need_detach:
        rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1).detach()
    else:
        rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1)
    rel_pose_skeleton[:, 1:] = rel_pose_skeleton[:, 1:] - rel_pose_skeleton[:, parents[1:]].clone()
    rel_pose_skeleton[:, 0] = rel_rest_pose[:, 0]

    # the predicted final pose
    final_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1)
    final_pose_skeleton = final_pose_skeleton - final_pose_skeleton[:, 0:1] + rel_rest_pose[:, 0:1]

    # assert phis.dim() == 3
    phis = phis / (torch.norm(phis, dim=2, keepdim=True) + 1e-8)

    global_orient_mat = batch_get_pelvis_orient(
        rel_pose_skeleton.clone(), rel_rest_pose.clone(), parents, children, dtype)

    # rot_mat_chain = [global_orient_mat]
    # rot_mat_local = [global_orient_mat]
    # print(global_orient_mat)

    rot_mat_chain = torch.zeros((batch_size, 24, 3, 3), dtype=torch.float32, device=pose_skeleton.device)
    rot_mat_local = torch.zeros_like(rot_mat_chain)
    rot_mat_chain[:, 0] = global_orient_mat
    rot_mat_local[:, 0] = global_orient_mat

    rotmat_twist_list = torch.zeros((batch_size, 23, 3, 3), dtype=torch.float32, device=pose_skeleton.device)
    rotmat_swing_list = torch.zeros((batch_size, 24, 3, 3), dtype=torch.float32, device=pose_skeleton.device)
    rotmat_swing_list[:, 0] = global_orient_mat

    idx_levs = [
        [0],
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12, 13, 14],
        [15, 16, 17],
        [18, 19],
        [20, 21],
        [22, 23],
        [24, 25, 26, 27, 28]
    ]
    if leaf_thetas is not None:
        idx_levs = idx_levs[:-1]

    all_child_rest_loc = rel_rest_pose[:, children[1:24]]
    all_child_rest_norm = torch.norm(all_child_rest_loc, dim=2, keepdim=True)

    # Convert twist to rot_mat
    # (B, K, 3, 1)
    twist_axis = all_child_rest_loc / (all_child_rest_norm + 1e-8)
    # (B, K, 1, 1)
    rx, ry, rz = torch.split(twist_axis, 1, dim=2)
    zeros = torch.zeros((batch_size, 23, 1, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=2) \
        .view((batch_size, 23, 3, 3))
    ident = torch.eye(3, dtype=dtype, device=device).reshape(1, 1, 3, 3)
    # (B, K, 1, 1)
    cos, sin = torch.split(phis, 1, dim=2)
    cos = torch.unsqueeze(cos, dim=3)
    sin = torch.unsqueeze(sin, dim=3)
    all_rot_mat_twist = ident + sin * K + (1 - cos) * torch.matmul(K, K)

    '''
    rot_mat_local = scripted_calc_swing(
        parents, children,
        rot_mat_chain, rel_pose_skeleton, unit_rel_rest_pose,
        all_rot_mat_twist, rot_mat_local)
    '''
    for idx_lev in range(1, len(idx_levs)):

        indices = idx_levs[idx_lev]
        if idx_lev == len(idx_levs) - 1:
            # leaf nodes
            assert NotImplementedError
        else:
            len_indices = len(indices)
            # (B, K, 3, 1)
            child_final_loc = torch.matmul(
                rot_mat_chain[:, parents[indices]].transpose(2, 3),
                rel_pose_skeleton[:, children[indices]])  # rotate back

            unit_child_rest = unit_rel_rest_pose[:, children[indices]]
            # child_rest_loc = rel_rest_pose[:, children[indices]]
            # (B, K, 1, 1)
            child_final_norm = torch.norm(child_final_loc, dim=2, keepdim=True)
            # child_rest_norm = torch.norm(child_rest_loc, dim=2, keepdim=True)

            # # (B, K, 3, 1)
            # axis = torch.cross(child_rest_loc, child_final_loc, dim=2)
            # axis_norm = torch.norm(axis, dim=2, keepdim=True)

            # # (B, K, 1, 1)
            # cos = torch.sum(child_rest_loc * child_final_loc, dim=2, keepdim=True) / (child_rest_norm * child_final_norm + 1e-8)
            # sin = axis_norm / (child_rest_norm * child_final_norm + 1e-8)

            # # (B, K, 3, 1)
            # axis = axis / (axis_norm + 1e-8)
            unit_child_final = child_final_loc / (child_final_norm + 1e-8)
            # unit_child_rest = child_rest_loc / (child_rest_norm + 1e-8)

            axis = torch.cross(unit_child_rest, unit_child_final, dim=2)
            cos = torch.sum(unit_child_rest * unit_child_final, dim=2, keepdim=True)
            sin = torch.norm(axis, dim=2, keepdim=True)

            # (B, K, 3, 1)
            axis = axis / (sin + 1e-8)

            # Convert location revolve to rot_mat by rodrigues
            # (B, K, 1, 1)
            rx, ry, rz = torch.split(axis, 1, dim=2)
            zeros = torch.zeros((batch_size, len_indices, 1, 1), dtype=dtype, device=device)

            K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=2) \
                .view((batch_size, len_indices, 3, 3))
            ident = torch.eye(3, dtype=dtype, device=device).reshape(1, 1, 3, 3)
            rot_mat_loc = ident + sin * K + (1 - cos) * torch.matmul(K, K)
            rot_mat_twist = all_rot_mat_twist[:, [i - 1 for i in indices]]

            rot_mat = torch.matmul(rot_mat_loc, rot_mat_twist)

            rot_mat_chain[:, indices] = torch.matmul(
                rot_mat_chain[:, parents[indices]],
                rot_mat
            )
            rot_mat_local[:, indices] = rot_mat
            rotmat_swing_list[:, indices] = rot_mat_loc
            rotmat_twist_list[:, [i - 1 for i in indices]] = rot_mat_twist

    return rotmat_swing_list, rotmat_twist_list


def batch_inverse_kinematics_decompose_rot(
        rot_mats,
        rest_pose,
        children, parents, dtype=torch.float32,
        leaf_thetas=None, need_detach=True):
    """
    Applies a batch of inverse kinematics transfoirm to the joints

    Parameters
    ----------
    pose_skeleton : torch.tensor BxNx3
        Locations of estimated pose skeleton.
    global_orient : torch.tensor Bx1x3x3
        Tensor of global rotation matrices
    phis : torch.tensor BxNx2
        The rotation on bone axis parameters
    rest_pose : torch.tensor Bx(N+1)x3
        Locations of rest_pose. (Template Pose)
    children: dict
        The dictionary that describes the kinematic chidrens for the model
    parents : torch.tensor Bx(N+1)
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32

    Returns
    -------
    rot_mats: torch.tensor Bx(N+1)x3x3
        The rotation matrics of each joints
    rel_transforms : torch.tensor Bx(N+1)x4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """
    batch_size = rot_mats.shape[0]
    device = rot_mats.device

    rel_rest_pose = rest_pose.clone()
    rel_rest_pose[:, 1:] -= rest_pose[:, parents[1:]].clone()
    rel_rest_pose = torch.unsqueeze(rel_rest_pose, dim=-1)
    unit_rel_rest_pose = rel_rest_pose / torch.norm(rel_rest_pose, dim=2, keepdim=True)

    # rotate the T pose
    rotate_rest_pose = torch.zeros_like(rel_rest_pose)
    # set up the root
    rotate_rest_pose[:, 0] = rel_rest_pose[:, 0]

    # if need_detach:
    #     rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1).detach()
    # else:
    #     rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1)
    # rel_pose_skeleton[:, 1:] = rel_pose_skeleton[:, 1:] - rel_pose_skeleton[:, parents[1:]].clone()
    # rel_pose_skeleton[:, 0] = rel_rest_pose[:, 0]

    # the predicted final pose
    # final_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1)
    # final_pose_skeleton = final_pose_skeleton - final_pose_skeleton[:, 0:1] + rel_rest_pose[:, 0:1]

    global_orient_mat = rot_mats[:, 0]

    rot_mat_chain = torch.zeros((batch_size, 24, 3, 3), dtype=torch.float32, device=rot_mats.device)
    rot_mat_local = torch.zeros_like(rot_mat_chain)
    rot_mat_chain[:, 0] = global_orient_mat
    rot_mat_local[:, 0] = global_orient_mat

    rotmat_twist_list = torch.zeros((batch_size, 23, 3, 3), dtype=torch.float32, device=rot_mats.device)
    angle_twist_list = torch.zeros((batch_size, 23, 1), dtype=torch.float32, device=rot_mats.device)
    rotmat_swing_list = torch.zeros((batch_size, 24, 3, 3), dtype=torch.float32, device=rot_mats.device)
    rotmat_swing_list[:, 0] = global_orient_mat

    idx_levs = [
        [0],
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12, 13, 14],
        [15, 16, 17],
        [18, 19],
        [20, 21],
        [22, 23],
        [24, 25, 26, 27, 28]
    ]

    for idx_lev in range(1, len(idx_levs)):

        indices = idx_levs[idx_lev]
        if idx_lev == len(idx_levs) - 1:
            # leaf nodes
            assert NotImplementedError
        else:
            len_indices = len(indices)
            # (B, K, 3, 1)

            child_final_loc = torch.matmul(
                rot_mats[:, indices],
                rel_rest_pose[:, children[indices]])  # rotate rest pose

            unit_child_rest = unit_rel_rest_pose[:, children[indices]]

            # (B, K, 1, 1)
            child_final_norm = torch.norm(child_final_loc, dim=2, keepdim=True)

            # # (B, K, 3, 1)
            # axis = axis / (axis_norm + 1e-8)
            unit_child_final = child_final_loc / (child_final_norm + 1e-8)
            # unit_child_rest = child_rest_loc / (child_rest_norm + 1e-8)

            axis = torch.cross(unit_child_rest, unit_child_final, dim=2)
            cos = torch.sum(unit_child_rest * unit_child_final, dim=2, keepdim=True)
            sin = torch.norm(axis, dim=2, keepdim=True)

            # (B, K, 3, 1)
            axis = axis / (sin + 1e-8)

            # Convert location revolve to rot_mat by rodrigues
            # (B, K, 1, 1)
            rx, ry, rz = torch.split(axis, 1, dim=2)
            zeros = torch.zeros((batch_size, len_indices, 1, 1), dtype=dtype, device=device)

            K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=2) \
                .view((batch_size, len_indices, 3, 3))
            ident = torch.eye(3, dtype=dtype, device=device).reshape(1, 1, 3, 3)
            rot_mat_swing = ident + sin * K + (1 - cos) * torch.matmul(K, K)

            rot_mat_twist = torch.matmul(
                rot_mat_swing.transpose(2, 3), rot_mats[:, indices]
            )
            twist_aa = roma.rotmat_to_rotvec(rot_mat_twist)
            twist_angle = torch.norm(twist_aa, dim=2, keepdim=True)
            twist_axis = twist_aa / twist_angle

            swing_axis = unit_child_rest.squeeze(dim=3)
            pos = torch.norm(swing_axis - twist_axis, dim=2, keepdim=True)
            neg = torch.norm(swing_axis + twist_axis, dim=2, keepdim=True)
            mask_inv = pos > neg
            twist_angle[mask_inv] = -1 * twist_angle[mask_inv]

            mask = twist_angle > math.pi
            twist_angle[mask] = twist_angle[mask] - 2 * math.pi
            mask = twist_angle < -math.pi
            twist_angle[mask] = twist_angle[mask] + 2 * math.pi

            max_norm = torch.maximum(pos, neg)
            assert (max_norm > 1.4).all(), (max_norm[max_norm <= 1.9])
            assert (max_norm > 1.4).all(), (max_norm, pos, neg, twist_angle)

            rotmat_twist_list[:, [i - 1 for i in indices]] = rot_mat_twist
            angle_twist_list[:, [i - 1 for i in indices]] = twist_angle

            rotmat_swing_list[:, indices] = rot_mat_swing
            rotmat_twist_list[:, [i - 1 for i in indices]] = rot_mat_twist

    return rotmat_swing_list, rotmat_twist_list, angle_twist_list

def batch_rot_composite(swing_rotmats, phis, rest_pose, children, parents, dtype=torch.float32):

    batch_size = swing_rotmats.shape[0]
    device = swing_rotmats.device

    rel_rest_pose = rest_pose.clone()
    rel_rest_pose[:, 1:] -= rest_pose[:, parents[1:]].clone()
    rel_rest_pose = torch.unsqueeze(rel_rest_pose, dim=-1)

    assert phis.dim() == 3
    phis = phis / (torch.norm(phis, dim=2, keepdim=True) + 1e-8)

    # torch.cuda.synchronize()
    # start_t = time.time()

    global_orient_mat = swing_rotmats[:, 0]

    rot_mat_chain = torch.zeros(
        (batch_size, 24, 3, 3), dtype=swing_rotmats.dtype, device=swing_rotmats.device)
    rot_mat_local = torch.zeros_like(rot_mat_chain)
    rot_mat_chain[:, 0] = global_orient_mat
    rot_mat_local[:, 0] = global_orient_mat

    idx_levs = [
        [0],
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12, 13, 14],
        [15, 16, 17],
        [18, 19],
        [20, 21],
        [22, 23],
        [24, 25, 26, 27, 28]
    ]

    for idx_lev in range(1, len(idx_levs)):

        indices = idx_levs[idx_lev]
        if idx_lev == len(idx_levs) - 1:
            # leaf nodes
            assert NotImplementedError
        else:
            len_indices = len(indices)

            child_rest_loc = rel_rest_pose[:, children[indices]]  # need rotation back ?
            # (B, K, 1, 1)
            child_rest_norm = torch.norm(child_rest_loc, dim=2, keepdim=True)

            rot_mat_swing = swing_rotmats[:, indices]
            # Convert spin to rot_mat
            # (B, K, 3, 1)
            spin_axis = child_rest_loc / (child_rest_norm + 1e-8)
            # (B, K, 1, 1)
            rx, ry, rz = torch.split(spin_axis, 1, dim=2)
            zeros = torch.zeros((batch_size, len_indices, 1, 1), dtype=dtype, device=device)
            K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=2) \
                .view((batch_size, len_indices, 3, 3))
            ident = torch.eye(3, dtype=dtype, device=device).reshape(1, 1, 3, 3)
            # (B, K, 1, 1)
            phi_indices = [item - 1 for item in indices]

            cos, sin = torch.split(phis[:, phi_indices], 1, dim=2)
            cos = torch.unsqueeze(cos, dim=3)
            sin = torch.unsqueeze(sin, dim=3)
            rot_mat_twist = ident + sin * K + (1 - cos) * torch.matmul(K, K)
            # rot_mat_twist = phis[:, phi_indices]
            rot_mat = torch.matmul(rot_mat_swing, rot_mat_twist)

            rot_mat_chain[:, indices] = torch.matmul(
                rot_mat_chain[:, parents[indices]],
                rot_mat
            )
            rot_mat_local[:, indices] = rot_mat

            if not (torch.det(rot_mat) > 0).all():
                idx = ~(torch.det(rot_mat) > 0)  # all 0 except the error batch
                idx = (idx.reshape(batch_size, -1).sum(dim=1) > 0)
                print(
                    'composite',
                    -1, idx_lev, rot_mat_swing[idx], rot_mat_twist[idx],
                    (torch.det(rot_mat_swing) > 0).all(),
                    (torch.det(rot_mat_twist) > 0).all(),
                    sin[idx], cos[idx], K[idx]
                )

                raise NotImplementedError

    # (B, K + 1, 3, 3)
    rot_mats = rot_mat_local

    return rot_mats


def calc_swing(
        parents: torch.Tensor, children: torch.Tensor,
        rot_mat_chain: torch.Tensor, rel_pose_skeleton: torch.Tensor, unit_rel_rest_pose: torch.Tensor,
        all_rot_mat_twist: torch.Tensor, rot_mat_local: torch.Tensor) -> torch.Tensor:

    idx_levs = [
        [0],
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12, 13, 14],
        [15, 16, 17],
        [18, 19],
        [20, 21],
        [22, 23],
    ]

    for idx_lev in range(1, len(idx_levs)):

        indices = idx_levs[idx_lev]

        len_indices = len(indices)
        # (B, K, 3, 1)
        child_final_loc = torch.matmul(
            rot_mat_chain[:, parents[indices]].transpose(2, 3),
            rel_pose_skeleton[:, children[indices]])  # rotate back

        unit_child_rest = unit_rel_rest_pose[:, children[indices]]
        # child_rest_loc = rel_rest_pose[:, children[indices]]
        # (B, K, 1, 1)
        child_final_norm = torch.norm(child_final_loc, dim=2, keepdim=True)
        # child_rest_norm = torch.norm(child_rest_loc, dim=2, keepdim=True)

        '''
        # (B, K, 3, 1)
        axis = torch.cross(child_rest_loc, child_final_loc, dim=2)
        axis_norm = torch.norm(axis, dim=2, keepdim=True)

        # (B, K, 1, 1)
        cos = torch.sum(child_rest_loc * child_final_loc, dim=2, keepdim=True) / (child_rest_norm * child_final_norm + 1e-8)
        sin = axis_norm / (child_rest_norm * child_final_norm + 1e-8)

        # (B, K, 3, 1)
        axis = axis / (axis_norm + 1e-8)
        '''
        unit_child_final = child_final_loc / (child_final_norm + 1e-8)
        # unit_child_rest = child_rest_loc / (child_rest_norm + 1e-8)

        axis = torch.cross(unit_child_rest, unit_child_final, dim=2)
        cos = torch.sum(unit_child_rest * unit_child_final, dim=2, keepdim=True)
        sin = torch.norm(axis, dim=2, keepdim=True)

        # (B, K, 3, 1)
        axis = axis / (sin + 1e-8)

        # Convert location revolve to rot_mat by rodrigues
        # (B, K, 1, 1)
        rx, ry, rz = torch.split(axis, 1, dim=2)
        # zeros = torch.zeros((batch_size, len_indices, 1, 1), dtype=dtype, device=device)
        zeros = torch.zeros_like(rx)

        K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=2).view((-1, len_indices, 3, 3))
        ident = torch.eye(3, dtype=rx.dtype, device=rx.device).reshape(1, 1, 3, 3)
        rot_mat_loc = ident + sin * K + (1 - cos) * torch.matmul(K, K)
        rot_mat_twist = all_rot_mat_twist[:, [i - 1 for i in indices]]

        rot_mat = torch.matmul(rot_mat_loc, rot_mat_twist)

        rot_mat_chain[:, indices] = torch.matmul(
            rot_mat_chain[:, parents[indices]],
            rot_mat
        )
        rot_mat_local[:, indices] = rot_mat

    return rot_mat_local



def ts_decompose(
        betas, pose_skeleton, phis,
        v_template, shapedirs, posedirs, J_regressor, parents, children,
        lbs_weights, dtype=torch.float32):
    device = betas.device

    # 1. Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    # 2. Get the rest joints
    # NxJx3 array
    rest_J = vertices2joints(J_regressor, v_shaped)

    rest_J = torch.zeros((v_shaped.shape[0], 29, 3), dtype=dtype, device=device)
    rest_J[:, :24] = vertices2joints(J_regressor, v_shaped)

    leaf_number = [411, 2445, 5905, 3216, 6617]
    leaf_vertices = v_shaped[:, leaf_number].clone()
    rest_J[:, 24:] = leaf_vertices

    # 1. Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    rotmat_swing, rotmat_twist = batch_inverse_kinematics_decompose(
        pose_skeleton.clone(), phis.clone(),
        rest_J.clone(), children, parents, dtype=dtype)

    return rotmat_swing, rotmat_twist


def ts_decompose_rot(
        betas, rotmats,
        v_template, shapedirs, posedirs, J_regressor, parents, children,
        lbs_weights, dtype=torch.float32):
    device = betas.device

    # 1. Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    # 2. Get the rest joints
    # NxJx3 array
    rest_J = vertices2joints(J_regressor, v_shaped)

    rest_J = torch.zeros((v_shaped.shape[0], 29, 3), dtype=dtype, device=device)
    rest_J[:, :24] = vertices2joints(J_regressor, v_shaped)

    leaf_number = [411, 2445, 5905, 3216, 6617]
    leaf_vertices = v_shaped[:, leaf_number].clone()
    rest_J[:, 24:] = leaf_vertices

    # 1. Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    rotmat_swing, rotmat_twist, angle_twist = batch_inverse_kinematics_decompose_rot(
        rotmats.clone(), rest_J.clone(),
        children, parents, dtype=dtype)

    return rotmat_swing, rotmat_twist, angle_twist


def ts_compose(betas, swing_rotmats, phis,
               v_template, shapedirs, posedirs, J_regressor, J_regressor_h36m, parents, children,
               lbs_weights, dtype=torch.float32):
    batch_size = max(betas.shape[0], swing_rotmats.shape[0])
    device = betas.device

    # 1. Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    # 2. Get the rest joints
    # NxJx3 array
    rest_J = torch.zeros((v_shaped.shape[0], 29, 3), dtype=dtype, device=device)
    rest_J[:, :24] = vertices2joints(J_regressor, v_shaped)

    leaf_number = [411, 2445, 5905, 3216, 6617]
    leaf_vertices = v_shaped[:, leaf_number].clone()
    rest_J[:, 24:] = leaf_vertices

    # 3. Get the rotation matrics
    rot_mats = batch_rot_composite(
        swing_rotmats.clone(), phis.clone(),
        rest_J.clone(), children, parents, dtype=dtype)

    test_joints = True
    if test_joints:
        J_transformed, A = batch_rigid_transform(rot_mats, rest_J[:, :24].clone(), parents[:24], dtype=dtype)
    else:
        J_transformed = None

    # assert torch.mean(torch.abs(rotate_rest_pose - J_transformed)) < 1e-5
    # 4. Add pose blend shapes
    # rot_mats: N x (J + 1) x 3 x 3
    ident = torch.eye(3, dtype=dtype, device=device)
    pose_feature = (rot_mats[:, 1:] - ident).view([batch_size, -1])
    pose_offsets = torch.matmul(pose_feature, posedirs) \
        .view(batch_size, -1, 3)

    v_posed = pose_offsets + v_shaped.detach()

    # 5. Do skinning:
    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                               dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]
    J_from_verts_h36m = vertices2joints(J_regressor_h36m, verts)

    return verts, J_transformed, J_from_verts_h36m, rot_mats

