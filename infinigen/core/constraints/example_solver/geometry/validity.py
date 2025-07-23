# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Karhan Kayan

import logging

import gin
from shapely.geometry import MultiPolygon, Point, Polygon

import infinigen.core.constraints.constraint_language as cl
from infinigen.core import tags as t
from infinigen.core.constraints.constraint_language.util import (
    blender_objs_from_names,
    meshes_from_names,
    project_to_xy_poly,
)
from infinigen.core.constraints.evaluator.node_impl.trimesh_geometry import (
    any_touching,
    any_touching_expand,
    constrain_contact,
)
from infinigen.core.constraints.example_solver.geometry.stability import stable_against
from infinigen.core.constraints.example_solver.state_def import State
from infinigen.core.util import blender as butil

logger = logging.getLogger(__name__)


def check_pre_move_validity(scene, a, parent_dict, dx, dy):
    """ """
    parent = parent_dict[a]
    a_mesh = meshes_from_names(scene, a)[0]
    parent_mesh = meshes_from_names(scene, parent)[0]
    blender_objs_from_names(a)[0]

    # move a mesh by dx, dy and check if the projection of a_mesh is contained in parent_mesh
    # a_mesh.apply_transform(trimesh.transformations.compose_matrix(translate=[dx,dy,0]))
    a_poly = project_to_xy_poly(a_mesh)
    parent_poly = project_to_xy_poly(parent_mesh)
    centroid = a_poly.centroid
    new_centroid = Point([centroid.x + dx, centroid.y + dy])
    # plot
    # fig, ax = plt.subplots()
    # if isinstance(parent_poly, Polygon):
    #     x, y = parent_poly.exterior.xy
    #     ax.fill(x, y, alpha=0.5, fc='red', ec='black', label='Polygon b')
    # elif isinstance(parent_poly, MultiPolygon):
    #     for sub_poly in parent_poly.geoms:
    #         x, y = sub_poly.exterior.xy
    #         ax.fill(x, y, alpha=0.5, fc='red', ec='black', label='Polygon b')
    # ax.plot(centroid.x, centroid.y, 'o', color='black', label='Random point')
    # plt.show()
    # scene.show()

    if isinstance(a_poly, Polygon):
        if not parent_poly.contains(new_centroid):
            # print("not contained")
            return False
    elif isinstance(a_poly, MultiPolygon):
        for sub_poly in a_poly.geoms:
            if not parent_poly.contains(new_centroid):
                # print("not contained")
                return False

    return True


def all_relations_valid(state, name, use_initial=False, fix_pos=False):
    rels = state.objs[name].relations

    for i, relation_state in enumerate(rels):
        match relation_state.relation:
            case cl.StableAgainst(_child_tags, _parent_tags, _margin):
                if "OfficeChairFactory" in name:
                    a = 1
                res = stable_against(
                    state,
                    name,
                    relation_state,
                    use_initial=use_initial,
                    fix_pos=fix_pos,
                )
                # import pdb
                # pdb.set_trace()

                if res != True:
                    logger.debug(
                        f"{name} failed relation {i=}/{len(rels)} {relation_state.relation} on {relation_state.target_name}"
                    )
                    return False
                a = 1
            case _:
                raise TypeError(f"Unhandled {relation_state.relation}")

    return True


@gin.configurable
def check_post_move_validity(
    state: State,
    name: str,
    disable_collision_checking=False,
    visualize=False,
    expand_collision=False,
    return_touch=False,
    use_initial=False,
):  # MARK
    # collision_objs = []
    # for k, os in state.objs.items():
    #     print(os.obj.name)
    #     if k != name and t.Semantics.NoCollision not in os.tags:
    #         collision_objs.append(os.obj.name)
    scene = state.trimesh_scene
    objstate = state.objs[name]

    collision_objs = [
        os.obj.name
        for k, os in state.objs.items()
        if k != name and t.Semantics.NoCollision not in os.tags
    ]

    if len(collision_objs) == 0:
        return True
    # invisible_others()
    # bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
    # visible_others()
    # check relation
    if "ArmChairFactory" in name:
        a = 1
    if name == "5048934_BookStackFactory":
        a = 1
    if not all_relations_valid(state, name, use_initial=use_initial):
        print("all_relations_valid not valid ", name)
        if visualize:
            vis_obj = butil.copy(objstate.obj)
            vis_obj.name = f"validity_relations_fail_{name}"
        if return_touch:
            return False, None
        else:
            return False
    if "FloorLampFactory" in name:
        a = 1
    # check collision
    if disable_collision_checking:
        return True
    if t.Semantics.NoCollision in objstate.tags:
        if return_touch:
            return True, None
        else:
            return True
    # objstate.obj.location
    # Vector((2.1989827156066895, 12.716106414794922, 0.8305753469467163))
    # objstate.obj.rotation_euler
    # Euler((0.0, -0.0, 1.570796012878418), 'XYZ')
    # if "SimpleBookcaseFactory" in name:

    if expand_collision:
        touch = any_touching_expand(  # mark
            scene,
            objstate.obj.name,
            collision_objs,
            bvh_cache=state.bvh_cache,
        )

    else:
        touch = any_touching(  # mark
            scene, objstate.obj.name, collision_objs, bvh_cache=state.bvh_cache
        )
    if not constrain_contact(touch, should_touch=None, max_depth=0.0001):
        if visualize:
            vis_obj = butil.copy(objstate.obj)
            vis_obj.name = f"validity_contact_fail_{name}"

        contact_names = [
            [x for x in t.names if not x.startswith("_")] for t in touch.contacts
        ]
        logger.debug(
            f"validity failed - {name} touched {contact_names[0]} {len(contact_names)=}"
        )
        if return_touch:
            return False, touch
        else:
            return False

    # available = path_to_door(  # mark
    #     scene, objstate.obj.name, collision_objs, bvh_cache=state.bvh_cache
    # )

    # supposed to go through the consgraph here
    if return_touch:
        return False, touch
    else:
        return True


@gin.configurable
def move_for_relation_and_collision(
    state: State,
    name: str,
    disable_collision_checking=False,
    visualize=False,
    expand_collision=False,
    return_touch=False,
    use_initial=False,
):  # MARK
    # import pdb
    # pdb.set_trace()
    scene = state.trimesh_scene
    objstate = state.objs[name]

    # collision_objs = [
    #     os.obj.name
    #     for k, os in state.objs.items()
    #     if k != name and t.Semantics.NoCollision not in os.tags
    # ]
    from infinigen_examples.steps.evaluate import get_relation_mapping
    from infinigen_examples.steps.tools import export_relation

    map_cp = get_relation_mapping(state)
    collision_objs = []
    for k, os in state.objs.items():
        if k != name and t.Semantics.NoCollision not in os.tags:
            objname = os.obj.name
            if hasattr(os, "populate_obj") and map_cp[k]["child"] != []:
                for child in map_cp[k]["child"]:
                    for rel in state.objs[child].relations:
                        if (
                            rel.target_name == k
                            and export_relation(rel.relation) == "on"
                        ):
                            objname = os.populate_obj
                            break
            collision_objs.append(objname)

    a_col_name = objstate.obj.name
    if hasattr(objstate, "populate_obj") and map_cp[name]["child"] != []:
        for child in map_cp[name]["child"]:
            for rel in state.objs[child].relations:
                if rel.target_name == name and export_relation(rel.relation) == "on":
                    a_col_name = objstate.populate_obj
                    break

    if len(collision_objs) == 0:
        return True
    # invisible_others()
    # bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
    # visible_others()
    # check relation

    if not all_relations_valid(state, name, use_initial=use_initial):
        print("all_relations_valid not valid ", name)
        if visualize:
            vis_obj = butil.copy(objstate.obj)
            vis_obj.name = f"validity_relations_fail_{name}"
        # if return_touch:
        #     return False, None
        # else:
        #     return False

    # check collision
    if disable_collision_checking:
        return True
    if t.Semantics.NoCollision in objstate.tags:
        if return_touch:
            return True, None
        else:
            return True
    # objstate.obj.location
    # Vector((2.1989827156066895, 12.716106414794922, 0.8305753469467163))
    # objstate.obj.rotation_euler
    # Euler((0.0, -0.0, 1.570796012878418), 'XYZ')
    # if "SimpleBookcaseFactory" in name:

    if expand_collision:
        touch = any_touching_expand(  # mark
            scene,
            a_col_name,  # objstate.obj.name,
            collision_objs,
            bvh_cache=state.bvh_cache,
        )

    else:
        touch = any_touching(  # mark
            scene, a_col_name, collision_objs, bvh_cache=state.bvh_cache
        )
    if not constrain_contact(touch, should_touch=None, max_depth=0.0001):
        if visualize:
            vis_obj = butil.copy(objstate.obj)
            vis_obj.name = f"validity_contact_fail_{name}"

        contact_names = [
            [x for x in t.names if not x.startswith("_")] for t in touch.contacts
        ]
        logger.debug(
            f"validity failed - {name} touched {contact_names[0]} {len(contact_names)=}"
        )
        if return_touch:
            return False, touch
        else:
            return False

    # available = path_to_door(  # mark
    #     scene, objstate.obj.name, collision_objs, bvh_cache=state.bvh_cache
    # )

    # supposed to go through the consgraph here
    if return_touch:
        return False, touch
    else:
        return True


# @gin.configurable
# def check_post_move_validity(
#     state: State, name: str, disable_collision_checking=False, visualize=False
# ):  # MARK
#     scene = state.trimesh_scene
#     objstate = state.objs[name]

#     collision_objs = [
#         os.obj.name
#         for k, os in state.objs.items()
#         if k != name and t.Semantics.NoCollision not in os.tags
#     ]

#     if len(collision_objs) == 0:
#         return True

#     if not all_relations_valid(state, name):
#         if visualize:
#             vis_obj = butil.copy(objstate.obj)
#             vis_obj.name = f"validity_relations_fail_{name}"

#         return False

#     if disable_collision_checking:
#         return True
#     if t.Semantics.NoCollision in objstate.tags:
#         return True

#     touch = any_touching(  # mark
#         scene, objstate.obj.name, collision_objs, bvh_cache=state.bvh_cache
#     )
#     if not constrain_contact(touch, should_touch=None, max_depth=0.0001):
#         if visualize:
#             vis_obj = butil.copy(objstate.obj)
#             vis_obj.name = f"validity_contact_fail_{name}"

#         contact_names = [
#             [x for x in t.names if not x.startswith("_")] for t in touch.contacts
#         ]
#         logger.debug(
#             f"validity failed - {name} touched {contact_names[0]} {len(contact_names)=}"
#         )
#         return False

#     # available = path_to_door(  # mark
#     #     scene, objstate.obj.name, collision_objs, bvh_cache=state.bvh_cache
#     # )

#     # supposed to go through the consgraph here
#     return True
