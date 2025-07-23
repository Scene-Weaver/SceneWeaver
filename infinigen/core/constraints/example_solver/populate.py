# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Alexander Raistrick: populate_state_placeholders, apply_cutter
# - Stamatis Alexandropoulos: Initial version of window cutting

import logging

import bpy
from tqdm import tqdm

from infinigen.core import tagging
from infinigen.core import tags as t
from infinigen.core.constraints import usage_lookup
from infinigen.core.constraints.constraint_language.util import delete_obj, sync_trimesh
from infinigen.core.constraints.example_solver.geometry import parse_scene
from infinigen.core.constraints.example_solver.state_def import State
from infinigen.core.placement.placement import parse_asset_name
from infinigen.core.util import blender as butil

logger = logging.getLogger(__name__)


def apply_cutter(state, objkey, cutter):
    os = state.objs[objkey]

    cut_objs = []
    for i, relation_state in enumerate(os.relations):
        # TODO in theory we maybe should check if they actually intersect
        # if relation_state.target_name == "newroom_0-0":
        #     a = 1
        #     continue
        parent_obj = state.objs[relation_state.target_name].obj
        butil.modify_mesh(
            parent_obj,
            "BOOLEAN",
            object=butil.copy(cutter),
            operation="DIFFERENCE",
            solver="FAST",
        )

        target_obj_name = state.objs[relation_state.target_name].obj.name
        cut_objs.append((relation_state.target_name, target_obj_name))

    cutter_col = butil.get_collection("placeholders:asset_cutters")
    butil.put_in_collection(cutter, cutter_col)

    return cut_objs


def populate_state_placeholders(state: State, filter=None, final=True):
    # 记录信息，表示正在填充占位符，并记录 final 和 filter 参数的值
    logger.info(f"Populating placeholders {final=} {filter=}")
    # 获取名为 "unique_assets" 的集合，用于存放唯一的资产对象
    unique_assets = butil.get_collection("unique_assets")
    unique_assets.hide_viewport = True
    # 如果 final 参数为 True，执行以下操作
    if final:
        for os in state.objs.values():
            # 如果对象属于房间类别，则更新对象的名字
            if t.Semantics.Room in os.tags:
                if os.obj.name.endswith(".meshed"):
                    continue
                os.obj = bpy.data.objects[os.obj.name + ".meshed"]

    targets = []  # 用于存放待处理的目标对象
    # 遍历状态中的所有对象
    for objkey, os in state.objs.items():
        if objkey == "8634849_Blackboard":
            a = 1
        # 如果对象没有生成器，则跳过
        if os.generator is None:
            continue
        # 如果提供了过滤条件，且当前对象不符合过滤条件，则跳过
        if filter is not None and not usage_lookup.has_usage(
            os.generator.__class__, filter
        ):
            continue
        # 如果对象名包含 "spawn_asset" 则说明已经处理过，跳过此对象
        if "spawn_asset" in os.obj.name:
            butil.put_in_collection(os.obj, unique_assets)
            logger.debug(f"Found already populated asset {os.obj.name=}, continuing")
            continue
        # 将当前对象的 objkey 加入目标列表
        targets.append(objkey)

    update_state_mesh_objs = []  # 用于存放需要更新的网格对象信息
    # 遍历目标对象，执行生成和处理
    for i, objkey in enumerate(targets):
        if objkey == "8634849_Blackboard":
            a = 1
        os = state.objs[objkey]
        placeholder = os.obj  # 获取占位符对象

        logger.info(f"Populating {i}/{len(targets)} {placeholder.name=}")

        old_objname = placeholder.name  # 记录原对象的名称
        update_state_mesh_objs.append((objkey, old_objname))  # 将旧对象信息加入更新列表

        *_, inst_seed = parse_asset_name(placeholder.name)  # 解析资产名称并提取实例种子
        # print(placeholder.name)
        if "BookStackFactory" in placeholder.name:
            a = 1

        # 使用生成器生成新的对象，并设置位置、旋转等属性
        os.obj = os.generator.spawn_asset(
            i=int(inst_seed),
            loc=placeholder.location,  # we could use placeholder=pholder here, but I worry pholder may have been modified
            rot=placeholder.rotation_euler,  # MARK
        )
        if os.size is not None:
            from infinigen.core.constraints.example_solver.moves.addition import (
                resize_obj,
            )

            os.obj = resize_obj(os.obj, os.size, apply_transform=False)
        os.generator.finalize_assets([os.obj])  # 完成生成器资产的最终处理
        butil.put_in_collection(os.obj, unique_assets)  # 将生成的对象放入唯一资产集合中

        # 查找可能存在的切割器（cutter），如果找到则应用切割操作
        cutter = next(
            (o for o in butil.iter_object_tree(os.obj) if o.name.endswith(".cutter")),
            None,
        )
        logger.debug(
            f"{populate_state_placeholders.__name__} found {cutter=} for {os.obj.name=}"
        )
        if cutter is not None:
            # 如果找到了切割器，则应用切割并记录切割的对象
            cut_objs = apply_cutter(state, objkey, cutter)
            logger.debug(
                f"{populate_state_placeholders.__name__} cut {cutter.name=} from {cut_objs=}"
            )
            update_state_mesh_objs += cut_objs  # 更新网格对象列表

    unique_assets.hide_viewport = False  # 恢复显示资产集合的视图
    # 如果是最终的处理，则返回
    if final:
        return
    # 否则，更新修改过的对象到 trimesh 状态中
    # objects modified in any way (via pholder update or boolean cut) must be synched with trimesh state
    for objkey, old_objname in tqdm(
        set(update_state_mesh_objs), desc="Updating trimesh with populated objects"
    ):
        os = state.objs[objkey]
        # 删除旧的 trimesh 对象
        # delete old trimesh
        delete_obj(state.trimesh_scene, old_objname, delete_blender=False)
        # 将新的生成的对象添加到 trimesh 场景中
        # put the new, populated object into the state
        parse_scene.preprocess_obj(os.obj)
        if not final:
            tagging.tag_canonical_surfaces(os.obj)  # 标记标准表面
        parse_scene.add_to_scene(
            state.trimesh_scene, os.obj, preprocess=True
        )  # 添加到场景


def populate_state_placeholders_mid(
    state: State, filter=None, final=False, update_trimesh=True
):
    # 记录信息，表示正在填充占位符，并记录 final 和 filter 参数的值
    logger.info(f"Populating placeholders {final=} {filter=}")
    # 获取名为 "unique_assets" 的集合，用于存放唯一的资产对象
    unique_assets = butil.get_collection("unique_assets")
    unique_assets.hide_viewport = True
    # 如果 final 参数为 True，执行以下操作
    targets = []  # 用于存放待处理的目标对象
    # 遍历状态中的所有对象

    for objkey, os in state.objs.items():
        # 如果对象没有生成器，则跳过
        if os.generator is None:
            continue
        # 如果提供了过滤条件，且当前对象不符合过滤条件，则跳过
        if filter is not None and not usage_lookup.has_usage(
            os.generator.__class__, filter
        ):
            continue
        # 如果对象名包含 "spawn_asset" 则说明已经处理过，跳过此对象
        # if hasattr(os, "populate_obj"): #TODO YYD
        #     continue
        if "spawn_asset" in os.obj.name:
            butil.put_in_collection(os.obj, unique_assets)
            logger.debug(f"Found already populated asset {os.obj.name=}, continuing")
            continue
        # 将当前对象的 objkey 加入目标列表
        targets.append(objkey)
    # save_path = "debug.blend"
    # bpy.ops.wm.save_as_mainfile(filepath=save_path)
    update_state_mesh_objs = []  # 用于存放需要更新的网格对象信息
    # 遍历目标对象，执行生成和处理
    for i, objkey in enumerate(targets):
        #     print("aaaaaaaaaaa")
        # pdb.set_trace()
        os = state.objs[objkey]
        placeholder = os.obj  # 获取占位符对象
        # if any(obj.name == placeholder.name.replace("bbox_placeholder","spawn_asset") for obj in unique_assets.objects):
        #     continue #TODO YYD

        logger.info(f"Populating {i}/{len(targets)} {placeholder.name=}")
        #'ThreedFrontCategoryFactory(2179127).bbox_placeholder(620454)'
        old_objname = placeholder.name  # 记录原对象的名称
        update_state_mesh_objs.append((objkey, old_objname))  # 将旧对象信息加入更新列表

        *_, inst_seed = parse_asset_name(placeholder.name)  # 解析资产名称并提取实例种子

        # # 使用生成器生成新的对象，并设置位置、旋转等属性 'ThreedFrontCategoryFactory(2179127).spawn_asset(620454)'
        populate_obj_name = placeholder.name.replace(
            "bbox_placeholder", "spawn_asset"
        ).replace("spawn_placeholder", "spawn_asset")

        if any(obj.name == populate_obj_name for obj in unique_assets.objects):
            obj = unique_assets.objects[populate_obj_name]
            obj.location = placeholder.location
            obj.rotation_mode = "XYZ"
            obj.rotation_euler = placeholder.rotation_euler

            scale_x = (
                placeholder.dimensions[0] / obj.dimensions[0]
                if obj.dimensions[0] != 0
                else 1
            )
            scale_y = (
                placeholder.dimensions[1] / obj.dimensions[1]
                if obj.dimensions[1] != 0
                else 1
            )
            scale_z = (
                placeholder.dimensions[2] / obj.dimensions[2]
                if obj.dimensions[2] != 0
                else 1
            )
            obj.scale = (scale_x, scale_y, scale_z)
            bpy.context.view_layer.objects.active = obj  # Set as active object
            obj.select_set(True)  # Select the object
            # 更新当前视图层，以反映对对象的修改
            bpy.context.view_layer.update()
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

        else:
            if inst_seed == "9128922":
                a = 1
            obj = os.generator.spawn_asset(  # TODO
                i=int(inst_seed),
                # placeholder=placeholder,
                loc=placeholder.location,  # we could use placeholder=pholder here, but I worry pholder may have been modified
                rot=placeholder.rotation_euler,  # MARK
            )
            # aa = bpy.data.objects.get("ObjaverseCategoryFactory(5444696).spawn_asset(3340520)")
            # if aa is not None:
            #     print(aa.rotation_euler)
            placeholder.name = old_objname
            # # 使用生成器生成新的对象，并设置位置、旋转等属性
            # os.obj = os.generator.spawn_asset(
            #     i=int(inst_seed),
            #     loc=placeholder.location,  # we could use placeholder=pholder here, but I worry pholder may have been modified
            #     rot=placeholder.rotation_euler,  # MARK
            # )
            # obj = os.obj
            if os.size is not None:
                from infinigen.core.constraints.example_solver.moves.addition import (
                    resize_obj,
                )

                obj = resize_obj(obj, os.size, apply_transform=False)
            os.generator.finalize_assets([obj])  # 完成生成器资产的最终处理
            butil.put_in_collection(
                obj, unique_assets
            )  # 将生成的对象放入唯一资产集合中
        bpy.context.view_layer.update()
        # 查找可能存在的切割器（cutter），如果找到则应用切割操作
        cutter = next(
            (o for o in butil.iter_object_tree(obj) if o.name.endswith(".cutter")),
            None,
        )
        logger.debug(
            f"{populate_state_placeholders.__name__} found {cutter=} for {os.obj.name=}"
        )
        if cutter is not None:
            # 如果找到了切割器，则应用切割并记录切割的对象
            cut_objs = apply_cutter(state, objkey, cutter)
            logger.debug(
                f"{populate_state_placeholders.__name__} cut {cutter.name=} from {cut_objs=}"
            )
            update_state_mesh_objs += cut_objs  # 更新网格对象列表
        state.objs[objkey].populate_obj = obj.name

    unique_assets.hide_viewport = False  # 恢复显示资产集合的视图
    # 如果是最终的处理，则返回
    if final:
        return

    # if update_trimesh:
    #     # 否则，更新修改过的对象到 trimesh 状态中
    #     # objects modified in any way (via pholder update or boolean cut) must be synched with trimesh state
    #     for objkey, old_objname in tqdm(
    #         set(update_state_mesh_objs), desc="Updating trimesh with populated objects"
    #     ):
    #         os = state.objs[objkey]
    #         # 删除旧的 trimesh 对象
    #         # delete old trimesh
    #         delete_obj(state.trimesh_scene, old_objname, delete_blender=False)
    #         # 将新的生成的对象添加到 trimesh 场景中
    #         # put the new, populated object into the state
    #         parse_scene.preprocess_obj(os.obj)
    #         if not final:
    #             tagging.tag_canonical_surfaces(os.obj)  # 标记标准表面
    #         parse_scene.add_to_scene(   #placeholder
    #             state.trimesh_scene, os.obj, preprocess=True
    #         )  # 添加到场景
    if update_trimesh:
        # 否则，更新修改过的对象到 trimesh 状态中
        # objects modified in any way (via pholder update or boolean cut) must be synched with trimesh state
        for objkey, old_objname in tqdm(
            set(update_state_mesh_objs), desc="Updating trimesh with populated objects"
        ):
            if objkey == "newroom_0-0":
                continue
            # populated obj
            os = state.objs[objkey]
            obj = bpy.data.objects.get(os.populate_obj)

            # 删除旧的 trimesh 对象
            # delete old trimesh
            delete_obj(state.trimesh_scene, obj.name, delete_blender=False)
            # 将新的生成的对象添加到 trimesh 场景中
            # put the new, populated object into the state
            parse_scene.preprocess_obj(obj)
            if not final:
                tagging.tag_canonical_surfaces(obj)  # 标记标准表面

            parse_scene.add_to_scene(
                state.trimesh_scene, obj, preprocess=True
            )  # 添加到场景
            # print("\n".join(state.trimesh_scene.geometry.keys()))
    else:
        # 否则，更新修改过的对象到 trimesh 状态中
        # objects modified in any way (via pholder update or boolean cut) must be synched with trimesh state
        for objkey, old_objname in tqdm(
            set(update_state_mesh_objs), desc="Updating trimesh with populated objects"
        ):
            if objkey == "newroom_0-0":
                continue
            os = state.objs[objkey]
            obj = bpy.data.objects.get(os.populate_obj)

            bpy.context.view_layer.update()
            # 应用变换，只对对象的缩放进行应用，不改变位置和旋转
            butil.apply_transform(obj, loc=False, rot=False, scale=True)
            sync_trimesh(state.trimesh_scene, obj.name)


def update_asset_location(state: State, filter=None, final=False, update_trimesh=True):
    # 记录信息，表示正在填充占位符，并记录 final 和 filter 参数的值
    logger.info(f"Populating placeholders {final=} {filter=}")
    # 获取名为 "unique_assets" 的集合，用于存放唯一的资产对象
    unique_assets = butil.get_collection("unique_assets")
    unique_assets.hide_viewport = True
    # 如果 final 参数为 True，执行以下操作
    targets = []  # 用于存放待处理的目标对象
    # 遍历状态中的所有对象

    for objkey, os in state.objs.items():
        # 如果对象没有生成器，则跳过
        if os.generator is None:
            continue
        # 如果提供了过滤条件，且当前对象不符合过滤条件，则跳过
        if filter is not None and not usage_lookup.has_usage(
            os.generator.__class__, filter
        ):
            continue
        # 如果对象名包含 "spawn_asset" 则说明已经处理过，跳过此对象
        # if hasattr(os, "populate_obj"): #TODO YYD
        #     continue
        if "spawn_asset" in os.obj.name:
            butil.put_in_collection(os.obj, unique_assets)
            logger.debug(f"Found already populated asset {os.obj.name=}, continuing")
            continue
        # 将当前对象的 objkey 加入目标列表
        targets.append(objkey)
    # save_path = "debug.blend"
    # bpy.ops.wm.save_as_mainfile(filepath=save_path)
    update_state_mesh_objs = []  # 用于存放需要更新的网格对象信息
    # 遍历目标对象，执行生成和处理
    for i, objkey in enumerate(targets):
        os = state.objs[objkey]
        placeholder = os.obj  # 获取占位符对象

        logger.info(f"Updating location of {i}/{len(targets)} {placeholder.name=}")
        #'ThreedFrontCategoryFactory(2179127).bbox_placeholder(620454)'
        old_objname = placeholder.name  # 记录原对象的名称
        update_state_mesh_objs.append((objkey, old_objname))  # 将旧对象信息加入更新列表

        *_, inst_seed = parse_asset_name(placeholder.name)  # 解析资产名称并提取实例种子

        # # 使用生成器生成新的对象，并设置位置、旋转等属性 'ThreedFrontCategoryFactory(2179127).spawn_asset(620454)'
        populate_obj_name = placeholder.name.replace(
            "bbox_placeholder", "spawn_asset"
        ).replace("spawn_placeholder", "spawn_asset")
        if any(obj.name == populate_obj_name for obj in unique_assets.objects):
            obj = unique_assets.objects[populate_obj_name]
            obj.location = placeholder.location
            obj.rotation_mode = "XYZ"
            obj.rotation_euler = placeholder.rotation_euler

            scale_x = (
                placeholder.dimensions[0] / obj.dimensions[0]
                if obj.dimensions[0] != 0
                else 1
            )
            scale_y = (
                placeholder.dimensions[1] / obj.dimensions[1]
                if obj.dimensions[1] != 0
                else 1
            )
            scale_z = (
                placeholder.dimensions[2] / obj.dimensions[2]
                if obj.dimensions[2] != 0
                else 1
            )
            obj.scale = (scale_x, scale_y, scale_z)
            bpy.context.view_layer.objects.active = obj  # Set as active object
            obj.select_set(True)  # Select the object
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        else:
            if inst_seed == "9128922":
                a = 1
            obj = os.generator.spawn_asset(  # TODO
                i=int(inst_seed),
                # placeholder=placeholder,
                loc=placeholder.location,  # we could use placeholder=pholder here, but I worry pholder may have been modified
                rot=placeholder.rotation_euler,  # MARK
            )
            # aa = bpy.data.objects.get("ObjaverseCategoryFactory(5444696).spawn_asset(3340520)")
            # if aa is not None:
            #     print(aa.rotation_euler)
            placeholder.name = old_objname
            # # 使用生成器生成新的对象，并设置位置、旋转等属性
            # os.obj = os.generator.spawn_asset(
            #     i=int(inst_seed),
            #     loc=placeholder.location,  # we could use placeholder=pholder here, but I worry pholder may have been modified
            #     rot=placeholder.rotation_euler,  # MARK
            # )
            # obj = os.obj
            if os.size is not None:
                from infinigen.core.constraints.example_solver.moves.addition import (
                    resize_obj,
                )

                obj = resize_obj(obj, os.size, apply_transform=False)
            os.generator.finalize_assets([obj])  # 完成生成器资产的最终处理
            butil.put_in_collection(
                obj, unique_assets
            )  # 将生成的对象放入唯一资产集合中
        bpy.context.view_layer.update()
        # 查找可能存在的切割器（cutter），如果找到则应用切割操作
        cutter = next(
            (o for o in butil.iter_object_tree(obj) if o.name.endswith(".cutter")),
            None,
        )
        logger.debug(
            f"{populate_state_placeholders.__name__} found {cutter=} for {os.obj.name=}"
        )
        if cutter is not None:
            # 如果找到了切割器，则应用切割并记录切割的对象
            cut_objs = apply_cutter(state, objkey, cutter)
            logger.debug(
                f"{populate_state_placeholders.__name__} cut {cutter.name=} from {cut_objs=}"
            )
            update_state_mesh_objs += cut_objs  # 更新网格对象列表
        state.objs[objkey].populate_obj = obj.name

    unique_assets.hide_viewport = False  # 恢复显示资产集合的视图
    # 如果是最终的处理，则返回
    if final:
        return

    # if update_trimesh:
    #     # 否则，更新修改过的对象到 trimesh 状态中
    #     # objects modified in any way (via pholder update or boolean cut) must be synched with trimesh state
    #     for objkey, old_objname in tqdm(
    #         set(update_state_mesh_objs), desc="Updating trimesh with populated objects"
    #     ):
    #         os = state.objs[objkey]
    #         # 删除旧的 trimesh 对象
    #         # delete old trimesh
    #         delete_obj(state.trimesh_scene, old_objname, delete_blender=False)
    #         # 将新的生成的对象添加到 trimesh 场景中
    #         # put the new, populated object into the state
    #         parse_scene.preprocess_obj(os.obj)
    #         if not final:
    #             tagging.tag_canonical_surfaces(os.obj)  # 标记标准表面
    #         parse_scene.add_to_scene(   #placeholder
    #             state.trimesh_scene, os.obj, preprocess=True
    #         )  # 添加到场景
    if update_trimesh:
        # 否则，更新修改过的对象到 trimesh 状态中
        # objects modified in any way (via pholder update or boolean cut) must be synched with trimesh state
        for objkey, old_objname in tqdm(
            set(update_state_mesh_objs), desc="Updating trimesh with populated objects"
        ):
            if objkey == "newroom_0-0":
                continue
            # populated obj
            os = state.objs[objkey]
            obj = bpy.data.objects.get(os.populate_obj)

            # 删除旧的 trimesh 对象
            # delete old trimesh
            delete_obj(state.trimesh_scene, obj.name, delete_blender=False)
            # 将新的生成的对象添加到 trimesh 场景中
            # put the new, populated object into the state
            parse_scene.preprocess_obj(obj)
            if not final:
                tagging.tag_canonical_surfaces(obj)  # 标记标准表面

            parse_scene.add_to_scene(
                state.trimesh_scene, obj, preprocess=True
            )  # 添加到场景
            # print("\n".join(state.trimesh_scene.geometry.keys()))
