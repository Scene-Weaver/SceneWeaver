import importlib
import os
import pickle
import types

import bpy
import dill
import trimesh
import trimesh.parent
import mathutils
from infinigen.assets.materials import invisible_to_camera
from infinigen.core import execute_tasks, init, placement, surface, tagging
from infinigen.core import tags as t
from infinigen.core.constraints.example_solver.room import decorate as room_dec
from infinigen.core.util import blender as butil
from infinigen_examples.util import constraint_util as cu
from infinigen_examples.util.generate_indoors_util import (
    apply_greedy_restriction,
    create_outdoor_backdrop,
    hide_other_rooms,
    place_cam_overhead,
    restrict_solving,
)
from infinigen_examples.util.visible import invisible_others, visible_others
from infinigen_examples.steps.draw_bbox import add_rotated_bbox_wireframe

def change_attr(obj, condition, replace_attr, visited=None, path=""):
    """
    递归遍历对象的所有属性，并返回符合条件的属性路径和值。

    :param obj: 要遍历的对象
    :param condition: 过滤条件（一个函数，如 lambda attr: isinstance(attr, int)）
    :param visited: 记录已经访问过的对象，防止循环引用
    :param path: 记录当前属性路径
    :return: 符合条件的属性列表 [(路径, 值)]
    """
    if visited is None:
        visited = set()
    if "infinigen.assets.materials.wear_tear" in path:
        a = 1
    # 避免重复访问对象（防止循环引用）
    obj_id = id(obj)
    if obj_id in visited:
        return []
    visited.add(obj_id)

    results = []

    # 遍历对象的 __dict__ 属性（如果有）
    if hasattr(obj, "__dict__"):
        for attr_name, attr_value in vars(obj).items():
            if attr_name == "material_params":
                a = 1
            attr_path = f"{path}.{attr_name}" if path else attr_name  # 记录完整路径
            if condition(attr_value):
                new_value = getattr(attr_value, replace_attr)
                setattr(obj, attr_name, new_value)
                results.append((attr_path, attr_value))  # 满足条件，加入结果
            elif isinstance(attr_value, (list, tuple)):  # 递归遍历可迭代对象
                for idx, item in enumerate(attr_value):
                    if condition(attr_value[idx]):
                        new_value = getattr(attr_value[idx], replace_attr)
                        attr_value[idx] = new_value
                        results.append((f"{attr_path}[{idx}]", attr_value))  #
                        continue
                    sub_path = f"{attr_path}[{idx}]"
                    results.extend(
                        change_attr(item, condition, replace_attr, visited, sub_path)
                    )
            elif isinstance(attr_value, dict):  # 递归遍历可迭代对象
                for k, item in attr_value.items():
                    if condition(attr_value[k]):
                        new_value = getattr(attr_value[k], replace_attr)
                        attr_value[k] = new_value
                        results.append((f"{attr_path}[{k}]", attr_value))
                        continue
                    sub_path = f"{attr_path}[{k}]"
                    results.extend(
                        change_attr(item, condition, replace_attr, visited, sub_path)
                    )
            else:
                results.extend(
                    change_attr(attr_value, condition, replace_attr, visited, attr_path)
                )  # 递归查找

    return results


def recover_attr(obj, condition, reconver_func, visited=None, path=""):
    if visited is None:
        visited = set()
    # 避免重复访问对象（防止循环引用）
    obj_id = id(obj)
    if obj_id in visited:
        return []

    results = []
    if (
        ".np." in path
        or ".sys." in path
        or "logging.Logger" in path
        or ".importlib." in path
        or ".butil." in path
        or "logging" in path
        or "/infinigen/" in path
        or ".t." in path
        or ".gin." in path
    ):
        return []

    try:
        flag = hasattr(obj, "__dict__")
    except:
        flag = False
    # 遍历对象的 __dict__ 属性（如果有）
    if flag:
        visited.add(obj_id)
        print(path)
        if "sofa_fabric" in path:
            a = 1
        d = vars(obj)
        lst = list(d.keys())
        # for attr_name, attr_value in vars(obj).items():
        for attr_name in lst:
            if attr_name == "__module__":
                continue
            attr_value = d[attr_name]

            if attr_name == "guard_surface":
                a = 1
            if (
                attr_name
                in [
                    "_globals",
                    "compat",
                    "sctypeDict",
                    "typecodes",
                    "_pytesttester",
                    "common",
                ]
                or attr_name.startswith("_")
                or "[" in attr_name
            ):
                continue
            attr_path = f"{path}.{attr_name}" if path else attr_name  # 记录完整路径
            if condition(attr_value):
                new_value = reconver_func(attr_value)
                setattr(obj, attr_name, new_value)
                results.append((attr_path, attr_value))  # 满足条件，加入结果
            elif isinstance(attr_value, (list, tuple)):  # 递归遍历可迭代对象
                for idx, item in enumerate(attr_value):
                    if condition(attr_value[idx]):
                        new_value = reconver_func(attr_value[idx])
                        try:
                            attr_value[idx] = new_value
                        except:
                            bpy.app.translations.register(idx, new_value)
                        results.append((f"{attr_path}[{idx}]", attr_value))
                        continue
                    sub_path = f"{attr_path}[{idx}]"
                    results.extend(
                        recover_attr(item, condition, reconver_func, visited, sub_path)
                    )
            elif isinstance(attr_value, dict):  # 递归遍历可迭代对象
                lst1 = list(attr_value.keys())
                # for k,item in attr_value.items():
                for k in lst1:
                    item = attr_value[k]
                    if condition(attr_value[k]):
                        new_value = reconver_func(attr_value[k])
                        attr_value[k] = new_value
                        results.append((f"{attr_path}[{k}]", attr_value))
                        continue
                    sub_path = f"{attr_path}[{k}]"
                    results.extend(
                        recover_attr(item, condition, reconver_func, visited, sub_path)
                    )
            else:
                results.extend(
                    recover_attr(
                        attr_value, condition, reconver_func, visited, attr_path
                    )
                )  # 递归查找

    return results


def export_relation(relation):
    child_tags = relation.child_tags
    parent_tags = relation.parent_tags
    margin = relation.margin

    if child_tags == cu.bottom and parent_tags == cu.floortags and margin == 0.01:
        relname = "onfloor"
    elif child_tags == cu.back and parent_tags == cu.walltags and margin == 0.07:
        relname = "against_wall"
    elif child_tags == cu.side and parent_tags == cu.walltags and margin == 0.05:
        relname = "side_against_wall"
    elif child_tags == cu.bottom and parent_tags == cu.top:
        relname = "ontop"
    elif child_tags == cu.bottom and parent_tags == {
        t.Subpart.SupportSurface,
        -t.Subpart.Top,
    }:
        relname = "on"
    elif child_tags == cu.front and parent_tags == cu.side4 and margin == 0.05: #side4
        relname = "front_against"
    elif child_tags == cu.front and parent_tags == cu.front and margin == 0.05:
        relname = "front_to_front"
    elif child_tags == cu.leftright and parent_tags == cu.leftright:
        relname = "leftright_leftright"
    elif child_tags == cu.side and parent_tags == cu.side:
        relname = "side_by_side"
    elif child_tags == cu.back and parent_tags == cu.back:
        relname = "back_to_back"
    else:
        a = 1

    return relname


def export_layout(state, solver, save_dir):
    import json

    results = dict()
    results["objects"] = dict()
    results["structure"] = dict()
    results["roomsize"] = [solver.dimensions[0], solver.dimensions[1]]
    children_map = dict()
    for objkey, objinfo in state.objs.items():
        if objkey.startswith("newroom_0-0"):
            continue
        elif objkey.startswith("window") or objkey.startswith("entrance"):
            results["structure"][objkey] = dict()
            results["structure"][objkey]["location"] = [
                round(a, 2) for a in list(objinfo.obj.location)
            ]
            results["structure"][objkey]["rotation"] = [
                round(a, 2) for a in list(objinfo.obj.rotation_euler)
            ]
            results["structure"][objkey]["size"] = [
                round(a, 2) for a in list(objinfo.obj.dimensions)
            ]
        else:
            offset_vector = calc_position_bias(objinfo.obj)
            results["objects"][objkey] = dict()
            results["objects"][objkey]["location"] = [
                round(a, 2) for a in list(objinfo.obj.location+offset_vector)
            ]
            results["objects"][objkey]["rotation"] = [
                round(a, 2) for a in list(objinfo.obj.rotation_euler)
            ]
            results["objects"][objkey]["size"] = [
                round(a, 2) for a in list(objinfo.obj.dimensions)
            ]
            if objinfo.relations is None:
                parent_relations = []
            else:
                parent_relations = [
                    [rel.target_name, export_relation(rel.relation)]
                    for rel in objinfo.relations
                ]
            results["objects"][objkey]["parent"] = parent_relations
            # parent_names = [rel.target_name for rel in objinfo.relations if rel.target_name!="newroom_0-0"]
            # for name in parent_names:
            #     if name not in children_map:
            #         children_map[name] = []
            #     children_map[name].append(objkey)
            # results["objects"][objkey]["parent"] = parent_names

    # for objkey in results["objects"]:
    #     results["objects"][objkey]["children"] = children_map[objkey] if objkey in children_map else []

    with open(save_dir, "w") as f:
        json.dump(results, f, indent=4)

def calc_position_bias(obj):

    # 获取 bounding box 在对象局部空间中的 8 个点
    bbox_corners = [mathutils.Vector(corner) for corner in obj.bound_box]

    # # 计算 bounding box 中心（局部坐标）
    # bbox_center_local = sum(bbox_corners, mathutils.Vector()) / 8 
    # 假设 bbox_corners 是 obj.bound_box 中的 8 个局部坐标点

    # 取底部的四个点（通常 Z 最小的四个）
    min_z = min(corner.z for corner in bbox_corners)
    bottom_corners = [corner for corner in bbox_corners if abs(corner.z - min_z) < 1e-4]

    # 计算底部中心（bottom center）
    bbox_bottom_center = sum(bottom_corners, mathutils.Vector()) / len(bottom_corners)

    # 转换为世界坐标
    bbox_center_world = obj.matrix_world @ bbox_bottom_center

    # 获取对象原点（世界坐标）
    origin_world = obj.location

    # 计算偏移向量（底部中心 - 原点）
    offset_vector = bbox_center_world - origin_world 

    print(obj.name)
    print("原点偏移向量 (world space):", offset_vector)
    print("X 偏移:", offset_vector.x)
    print("Y 偏移:", offset_vector.y)
    print("Z 偏移:", offset_vector.z)
    return offset_vector



def render_scene(p, solved_bbox, camera_rigs, state, filename="debug.jpg"):
    rooms_meshed = butil.get_collection("placeholders:room_meshes")
    rooms_split = room_dec.split_rooms(list(rooms_meshed.objects))

    def invisible_room_ceilings():
        rooms_split["exterior"].hide_viewport = True
        rooms_split["exterior"].hide_render = True
        rooms_split["ceiling"].hide_render = True
        invisible_to_camera.apply(list(rooms_split["ceiling"].objects))
        invisible_to_camera.apply(
            [o for o in bpy.data.objects if "CeilingLight" in o.name]
        )

    p.run_stage("invisible_room_ceilings", invisible_room_ceilings, use_chance=False)

    p.run_stage(
        "overhead_cam",
        place_cam_overhead,
        cam=camera_rigs[0],
        bbox=solved_bbox,
        use_chance=False,
    )

    # camera_rigs[0].rotation_euler = [0,0,1.57]
    bpy.context.scene.camera = camera_rigs[0]

    

    invisible_others(hide_placeholder=True)
    bpy.context.scene.render.resolution_x = 1920
    bpy.context.scene.render.resolution_y = 1080
    bpy.context.scene.render.image_settings.file_format = "JPEG"
    # bpy.context.scene.render.image_settings.color_mode = 'RGBA'  # Include alpha channel
    # bpy.context.scene.render.film_transparent = True  # For Cycles
    bpy.context.scene.render.filepath = os.path.join(filename)
    bpy.ops.render.render(write_still=True)
    visible_others()

    get_bbox(state)

    invisible_others(hide_all=True)
    # Set resolution
    bpy.context.scene.render.resolution_x = 1920
    bpy.context.scene.render.resolution_y = 1080
    # Use a file format that supports transparency
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'  # Include alpha channel
    # Enable transparency
    bpy.context.scene.render.film_transparent = True  # For Cycles
    filename_bbox = filename.replace(".jpg","_bbox.png")
    bpy.context.scene.render.filepath = os.path.join(filename_bbox)
    bpy.ops.render.render(write_still=True)
    visible_others(view_all=True)
    
    merge_two_image(filename,filename_bbox)

    # modified_output_path = bpy.path.abspath("render_8_coord.jpg")
    # world_to_image(filename, modified_output_path)

    bpy.context.scene.camera = None
    return

def merge_two_image(background_imgfile,foregroung_imgfile):
    from PIL import Image

    # Load base JPEG image
    jpeg_image = Image.open(background_imgfile).convert("RGB")

    # Load PNG with transparency
    png_image = Image.open(foregroung_imgfile).convert("RGBA")

    # Ensure both images are the same size (optional: resize PNG)
    png_image = png_image.resize(jpeg_image.size)

    # Convert JPEG to RGBA so it can handle alpha
    jpeg_rgba = jpeg_image.convert("RGBA")

    # Paste PNG on top with transparency
    combined = Image.alpha_composite(jpeg_rgba, png_image)

    # Save result
    filename = background_imgfile.replace(".jpg","_marked.jpg")
    # combined.save("combined_image.png")  # Save as PNG to preserve transparency
    combined.convert("RGB").save(filename, "JPEG", quality=95)

    return

def get_bbox(state):
    for name in state.objs:
        if name.startswith("window") or name == "newroom_0-0" or name == "entrance":
            continue
        obj = state.objs[name].obj
        cat_name = name.split("_")[1]
        if cat_name.endswith("Factory"):
            cat_name = cat_name[:-7]
        add_rotated_bbox_wireframe(obj,cat_name)
    save_path = "debug.blend"
    bpy.ops.wm.save_as_mainfile(filepath=save_path)
    return 

def world_to_image(image_path, output_path):
    import os

    import bpy_extras
    from mathutils import Vector
    from PIL import Image, ImageDraw, ImageFont

    def calc_point(x, y, z=0):
        world_coords = Vector([x, y, z])
        # Convert world coordinates to camera view space (normalized)
        co_2d = bpy_extras.object_utils.world_to_camera_view(scene, cam, world_coords)

        # Convert normalized coordinates to image pixel coordinates
        pixel_x = int(co_2d.x * res_x)
        pixel_y = int(
            (1 - co_2d.y) * res_y
        )  # Flip Y-axis (Blender's origin is bottom-left)
        print(f"3D World Coords (0,0,0) : {world_coords}")
        print(f"Projected 2D Image Coords: ({pixel_x}, {pixel_y})")

        draw.ellipse(
            [
                (pixel_x - dot_size, pixel_y - dot_size),
                (pixel_x + dot_size, pixel_y + dot_size),
            ],
            fill="red",
            outline="red",
        )

        # Draw the text label next to the point
        draw.text((pixel_x + 10, pixel_y - 10), f"({x}, {y})", fill="red", font=font)

        return

    # Get the scene and camera
    scene = bpy.context.scene
    cam = scene.camera.children[1]

    # Get render resolution and aspect ratio
    render = scene.render
    res_x = render.resolution_x * render.pixel_aspect_x
    res_y = render.resolution_y * render.pixel_aspect_y

    # Load the rendered image using PIL
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Try to load a font, otherwise use default
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default(30)

    # Draw a red dot at the calculated 2D coordinate
    dot_size = 5

    # for point in [(0,0),(10,12)]:
    #     calc_point(point[0],point[1])
    for x in range(0, 11, 2):
        for y in range(0, 13, 2):
            calc_point(x, y)

    # Save the modified image
    image.save(output_path)
    print(f"Image with marked point saved at {output_path}")


# def save_record(state,solver,stages,consgraph,iter=0):
def save_record(state, solver, terrain, house_bbox, solved_bbox, iter, p):
    # state.trimesh_scene = None
    save_path = f"record_files/scene_{iter}.blend"
    bpy.ops.wm.save_as_mainfile(filepath=save_path)
    
    for name in state.trimesh_scene.geometry.keys():
        state.trimesh_scene.geometry[name].fcl_obj = None
        state.trimesh_scene.geometry[name].col_obj = None

    state.bvh_cache = None

    for obj_name in state.objs.keys():
        # blender obj
        state.objs[obj_name].obj = state.objs[obj_name].obj.name
        # material
        # try:
        #     material = generator.material_params
        #     params = generator.params
        #     for mat in material.keys():
        #         material[mat] = material[mat].name
        #         params[mat] = material[mat]
        # except:
        #     pass

        # #generator
        # generator = state.objs[obj_name].generator
        # if generator is not None:
        #     for attr in dir(generator):
        #         m = getattr(generator,attr)
        #         if isinstance(m, types.ModuleType):
        #             setattr(generator, attr, m.__name__)
        #     if hasattr(generator, "base_factory") and generator.base_factory is not None:
        #         for attr in dir(generator.base_factory):
        #             m = getattr(generator.base_factory,attr)
        #             if isinstance(m, types.ModuleType):
        #                 setattr(generator.base_factory, attr, m.__name__)

        matches = change_attr(
            state.objs[obj_name].generator,
            lambda attr: isinstance(attr, types.ModuleType),
            "__name__",
        )
        matches = change_attr(
            state.objs[obj_name].generator,
            lambda attr: isinstance(attr, bpy.types.Material),
            "name",
        )
        matches = change_attr(
            state.objs[obj_name].generator,
            lambda attr: isinstance(attr, bpy.types.Collection),
            "name",
        )

    # for path, value in matches:
    # print(f"Found int at {path}: {value}")
    with open(f"record_files/state_{iter}.pkl", "wb") as file:
        # for obj_name in state.objs.keys():
        #     #blender obj
        #     try:
        #         dill.dump(state.objs[obj_name], file)
        #     except:
        #         import pdb
        #         pdb.set_trace()
        dill.dump(state, file)

    tagging.tag_system.save_tag()

    with open(f"record_files/solver_{iter}.pkl", "wb") as file:
        dill.dump(solver, file)

    with open(f"record_files/p_{iter}.pkl", "wb") as file:
        dill.dump(p, file)

    # with open(f"record_files/stages_{iter}.pkl", "wb") as file:
    #     pickle.dump(stages, file)

    # with open(f"record_files/consgraph_{iter}.pkl", "wb") as file:
    #     pickle.dump(consgraph, file)

    # with open(f"record_files/limits_{iter}.pkl", "wb") as file:
    #     pickle.dump(limits, file)

    with open(f"record_files/terrain_{iter}.pkl", "wb") as file:
        pickle.dump(terrain, file)

    # with open(f"record_files/solved_rooms_{iter}.pkl", "wb") as file:
    #     pickle.dump(solved_rooms, file)

    with open(f"record_files/house_bbox_{iter}.pkl", "wb") as file:
        pickle.dump(house_bbox, file)

    with open(f"record_files/solved_bbox_{iter}.pkl", "wb") as file:
        pickle.dump(solved_bbox, file)

    # with open(f"record_files/camera_rigs_{iter}.pkl", "wb") as file:
    #     pickle.dump(camera_rigs, file)

    env_file = f"record_files/env_{iter}.pkl"
    with open(env_file, "wb") as f:
        pickle.dump(dict(os.environ), f)

    for obj_name in state.objs.keys():
        state.objs[obj_name].obj = bpy.data.objects.get(state.objs[obj_name].obj)

    return


def is_module(attr):
    isstr = (
        isinstance(attr, str) and attr != "__module__" and attr.startswith("infinigen.")
    )
    if isstr:
        try:
            importlib.import_module(attr)
            return True
        except:
            return False


def is_material(attr):
    isstr = isinstance(attr, str)
    x = None
    if isstr:
        x = bpy.data.materials.get(attr)
    return x is not None


def is_collection(attr):
    isstr = isinstance(attr, str)
    x = None
    if isstr:
        x = bpy.data.collections.get(attr)
    return x is not None


def load_record(iter):
    with open(f"record_files/solver_{iter}.pkl", "rb") as file:
        solver = dill.load(file)

    # with open(f"record_files/stages_{iter}.pkl", "wb") as file:
    #     stages = pickle.load(file)

    # with open(f"record_files/consgraph_{iter}.pkl", "wb") as file:
    #     consgraph = pickle.load(file)

    # with open(f"record_files/limits_{iter}.pkl", "wb") as file:
    #     limits = pickle.load(file)

    with open(f"record_files/terrain_{iter}.pkl", "rb") as file:
        terrain = pickle.load(file)

    # with open(f"record_files/solved_rooms_{iter}.pkl", "wb") as file:
    #     solved_rooms = pickle.load(file)

    with open(f"record_files/house_bbox_{iter}.pkl", "rb") as file:
        house_bbox = pickle.load(file)

    with open(f"record_files/solved_bbox_{iter}.pkl", "rb") as file:
        solved_bbox = pickle.load(file)

    # with open(f"record_files/camera_rigs_{iter}.pkl", "wb") as file:
    #     camera_rigs = pickle.load(file)

    tagging.tag_system.load_tag()

    # with open(f"record_files/p_{iter}.pkl", "rb") as file:
    #     p = pickle.load(file)
    p = None

    save_path = f"record_files/scene_{iter}.blend"

    if not bpy.data.objects.get("newroom_0-0"):
        bpy.ops.wm.open_mainfile(filepath=save_path, load_ui=False, use_scripts=False)

    with open(f"record_files/state_{iter}.pkl", "rb") as file:
        state = dill.load(file)
    
    for obj_name in state.objs.keys():
        # blender obj
        state.objs[obj_name].obj = bpy.data.objects.get(
            state.objs[obj_name].obj
        )  # TODO YYD
        # if hasattr(state.objs[obj_name], "populate_obj"):
        #     state.objs[obj_name].obj = bpy.data.objects.get(state.objs[obj_name].populate_obj)
        # else:
        #     state.objs[obj_name].obj = bpy.data.objects.get(state.objs[obj_name].obj)

        matches = recover_attr(
            state.objs[obj_name].generator,
            is_module,
            lambda attr: importlib.import_module(attr),
        )
        matches = recover_attr(
            state.objs[obj_name].generator,
            is_material,
            lambda attr: bpy.data.materials.get(attr),
        )
        matches = recover_attr(
            state.objs[obj_name].generator,
            is_collection,
            lambda attr: bpy.data.collections.get(attr),
        )
        # #generator
        # generator = state.objs[obj_name].generator
        # if generator is not None:
        #     for attr in dir(generator):
        #         if attr=="__module__":
        #             continue
        #         module_name = getattr(generator,attr)
        #         try:
        #             m = importlib.import_module(module_name)
        #             setattr(generator, attr, m)
        #         except:
        #             pass
        #     if hasattr(generator, "base_factory") and generator.base_factory is not None:
        #         for attr in dir(generator.base_factory):
        #             if attr=="__module__":
        #                 continue
        #             module_name = getattr(generator.base_factory,attr)
        #             try:
        #                 m = importlib.import_module(module_name)
        #                 setattr(generator.base_factory, attr, m)
        #             except:
        #                 pass

        # #material
        # try:
        #     material = generator.material_params
        #     params = generator.params
        #     for mat in material.keys():
        #         m = bpy.data.materials.get(material[mat])
        #         material[mat] = m
        #         params[mat] = m
        # except:
        #     pass

    state.__post_init__()

    solver.state = state

    with open(f"record_files/env_{iter}.pkl", "rb") as f:
        env_vars = pickle.load(f)
    json_name = os.getenv("JSON_RESULTS")
    os.environ.update(env_vars)
    os.environ["JSON_RESULTS"] = json_name

    return state, solver, terrain, house_bbox, solved_bbox, p
