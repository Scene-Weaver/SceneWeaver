import copy

import mathutils
import numpy as np

D_base = 0.0
EXPAND_DISTANCE = {
    # front
    "LargeShelfFactory": [D_base, 0, 0],
    "LargeShelfBaseFactory": [D_base, 0, 0],
    # "LargePlantContainerFactory":[D_base,0,0],
    "KitchenSpaceFactory": [D_base, 0, 0],
    "KitchenCabinetBaseFactory": [D_base, 0, 0],
    "KitchenCabinetFactory": [D_base, 0, 0],
    "CabinetDrawerBaseFactory": [D_base, 0, 0],
    "TVStandFactory": [D_base, 0, 0],
    "CabinetDoorBaseFactory": [D_base, 0, 0],
    "CabinetDoorIkeaFactory": [D_base, 0, 0],
    "CabinetBaseFactory": [D_base, 0, 0],
    "CellShelfFactory": [D_base, 0, 0],
    "CellShelfBaseFactory": [D_base, 0, 0],
    "SimpleBookcaseFactory": [D_base, 0, 0],
    "SimpleBookcaseBaseFactory": [D_base, 0, 0],
    "SidetableDeskFactory": [D_base, 0, 0],
    "SimpleDeskFactory": [D_base, 0, 0],
    "SimpleDeskBaseFactory": [D_base, 0, 0],
    "SingleCabinetFactory": [D_base, 0, 0],
    "SingleCabinetBaseFactory": [D_base, 0, 0],
    "TriangleShelfBaseFactory": [D_base, 0, 0],
    "TriangleShelfFactory": [D_base, 0, 0],
    "ArmChairFactory": [D_base, 0, 0],
    "SofaFactory": [D_base, 0, 0],
    "BathroomSinkFactory": [D_base, 0, 0],
    "ToiletFactory": [D_base, 0, 0],
    "StandingSinkFactory": [D_base, 0, 0],
    "OvenFactory": [D_base, 0, 0],
    "BeverageFridgeFactory": [D_base, 0, 0],
    "DishwasherFactory": [D_base, 0, 0],
    "MicrowaveFactory": [D_base, 0, 0],
    # front,side
    "BedFactory": [D_base, 0, D_base],
    # "BedFrameFactory":[D_base,0,D_base],
    # front,back,side
    "KitchenIslandFactory": [D_base, D_base, 2 * D_base],
    "CountertopFactory": [D_base, D_base, 2 * D_base],
}


def get_expand_distance(name):
    for key in EXPAND_DISTANCE.keys():
        if name.startswith(key):
            # import pdb
            # pdb.set_trace()
            # print(name,EXPAND_DISTANCE[key])
            return EXPAND_DISTANCE[key]
    # print(name," not expand !!!!!!!")
    return [0, 0, 0]


def expand_mesh(geom, name):
    d_front, d_back, d_side = get_expand_distance(name)

    if d_front == 0 and d_back == 0 and d_side == 0:
        return geom

    T_old = geom.current_transform
    mesh_copy = copy.deepcopy(geom)
    mesh_copy.apply_transform(np.linalg.inv(T_old))
    vertices = mesh_copy.vertices
    ### move to proper pose for rescale
    # back expand = False
    back, front = vertices.min(0)[0], vertices.max(0)[0]
    left, right = vertices.min(0)[1], vertices.max(0)[1]
    v_center = [
        (back + front) / 2,  # back
        (left + right) / 2,  # left and right -> middle
        0,
    ]  # do not change z, so 0 is ok
    vertices -= v_center

    loc, rot, scale = mathutils.Matrix(T_old).decompose()
    ### rescale
    scaling_factor_front = (
        ((front - back) / 2 * scale[0] + d_front) / scale[0] / ((front - back) / 2)
    )
    scaling_factor_back = (
        ((front - back) / 2 * scale[0] + d_back) / scale[0] / ((front - back) / 2)
    )
    scaling_factor_side = (
        ((right - left) * scale[1] + d_side) / scale[1] / (right - left)
    )
    print(name, scaling_factor_front, scaling_factor_back, scaling_factor_side)

    scale_matrix = np.eye(3)

    scale_matrix[0, :] *= scaling_factor_front  # front

    vertices[vertices[:, 0] > 0] = vertices[vertices[:, 0] > 0] @ scale_matrix

    scale_matrix = np.eye(3)
    scale_matrix[0, :] *= scaling_factor_back  # back
    vertices[vertices[:, 0] < 0] = vertices[vertices[:, 0] < 0] @ scale_matrix

    scale_matrix = np.eye(3)
    scale_matrix[1, :] *= scaling_factor_side  # side
    vertices = vertices @ scale_matrix

    ### move back to original pose, finish expand
    vertices += v_center
    mesh_copy.vertices = vertices
    mesh_copy.apply_transform(T_old)
    return mesh_copy
