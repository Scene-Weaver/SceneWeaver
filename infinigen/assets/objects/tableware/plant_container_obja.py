# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
import bpy
import numpy as np
from numpy.random import uniform

import GPT
from GPT.constants import OBJATHOR_ASSETS_DIR
from infinigen.assets.material_assignments import AssetList
from infinigen.assets.objects.cactus import CactusFactory
from infinigen.assets.objects.monocot import MonocotFactory
from infinigen.assets.objects.mushroom import MushroomFactory
from infinigen.assets.objects.small_plants import (
    FernFactory,
    SnakePlantFactory,
    SpiderPlantFactory,
    SucculentFactory,
)
from infinigen.assets.objects.tableware.pot import PotFactory
from infinigen.assets.utils.decorate import (
    read_edge_center,
    read_edge_direction,
    remove_vertices,
    select_edges,
    subsurf,
)
from infinigen.assets.utils.object import join_objects, new_bbox, origin2lowest
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform


class PlantPotFactory(PotFactory):
    def __init__(self, factory_seed, coarse=False):
        super(PlantPotFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(self.factory_seed):
            self.has_handle = self.has_bar = self.has_guard = False
            self.depth = log_uniform(0.5, 1.0)
            self.r_expand = uniform(1.1, 1.3)
            alpha = uniform(0.5, 0.8)
            self.r_mid = (self.r_expand - 1) * alpha + 1
            material_assignments = AssetList["PlantContainerFactory"]()
            self.surface = material_assignments["surface"].assign_material()
            self.scale = log_uniform(0.08, 0.12)


class PlantContainerFactory(AssetFactory):
    is_fragile = False
    allow_transparent = False

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        self.retriever = GPT.Retriever

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        self.side_size = 0.1437744745933212
        self.hight = 0.6589937384939065 * 0.1131059492213359 + 0.43295466035780455
        return new_bbox(
            -self.side_size,
            self.side_size,
            -self.side_size,
            self.side_size,
            -0.02,
            self.hight,
        )

    def create_asset(self, **params) -> bpy.types.Object:
        from infinigen.assets.objaverse_assets.load_asset import load_pickled_3d_asset

        cat = "Plant Container"
        object_names = self.retriever.retrieve_object_by_cat(cat)
        for obj_name, score in object_names:
            basedir = OBJATHOR_ASSETS_DIR
            # indir = f"{basedir}/processed_2023_09_23_combine_scale"
            filename = f"{basedir}/{obj_name}/{obj_name}.pkl.gz"
            try:
                obj = load_pickled_3d_asset(filename)
                break
            except:
                continue
        scale = np.min(
            np.array([self.side_size, self.side_size, 0.43295466035780455])
            / np.max(np.abs(np.array(obj.bound_box)), 0)
        )
        obj.scale = [scale] * 3

        return obj


class PlantContainerFactoryOld(AssetFactory):
    plant_factories = [
        CactusFactory,
        MushroomFactory,
        FernFactory,
        SucculentFactory,
        SpiderPlantFactory,
        SnakePlantFactory,
    ]

    def __init__(self, factory_seed, coarse=False):
        super(PlantContainerFactoryOld, self).__init__(factory_seed, coarse)
        with FixedSeed(self.factory_seed):
            self.base_factory = PlantPotFactory(self.factory_seed, coarse)
            self.dirt_ratio = uniform(0.7, 0.8)
            material_assignments = AssetList["PlantContainerFactory"]()
            self.dirt_surface = material_assignments["dirt_surface"].assign_material()
            fn = np.random.choice(self.plant_factories)
            self.plant_factory = fn(self.factory_seed)
            self.side_size = self.base_factory.scale * self.base_factory.r_expand
            self.top_size = uniform(0.4, 0.6)

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        return new_bbox(
            -self.side_size,
            self.side_size,
            -self.side_size,
            self.side_size,
            -0.02,
            self.base_factory.depth * self.base_factory.scale + self.top_size,
        )

    def create_asset(self, i, **params) -> bpy.types.Object:
        # 使用 base_factory 创建基础资产对象
        obj = self.base_factory.create_asset(i=i, **params)
        # # 读取模型的边缘方向，并检查边缘是否接近平行于水平面
        horizontal = np.abs(read_edge_direction(obj)[:, -1]) < 0.1

        # 获取模型的边缘中心位置，并提取 z 坐标
        edge_center = read_edge_center(obj)
        z = edge_center[:, -1]

        # 计算脏物体（例如污垢或泥土）的 z 坐标，基于工厂的深度和比例因子
        dirt_z = self.dirt_ratio * self.base_factory.depth * self.base_factory.scale
        idx = np.argmin(np.abs(z - dirt_z) - horizontal * 10)
        # 计算边缘的半径，使用边缘中心坐标的 x 和 y 坐标来计算
        radius = np.sqrt((edge_center[idx] ** 2)[:2].sum())

        selection = np.zeros_like(z).astype(bool)
        selection[idx] = True
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_mode(type="EDGE")
            select_edges(obj, selection)
            bpy.ops.mesh.loop_multi_select(ring=False)
            bpy.ops.mesh.duplicate_move()
            bpy.ops.mesh.separate(type="SELECTED")

        # 获取新创建的脏物体（dirt_）
        dirt_ = bpy.context.selected_objects[-1]
        butil.select_none()  # 取消所有选择
        # 完成基础资产的最终处理
        self.base_factory.finalize_assets(obj)
        # 对脏物体进行编辑，填充网格并应用修改
        with butil.ViewportMode(dirt_, "EDIT"):
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.fill_grid()
        subsurf(dirt_, 3)
        self.dirt_surface.apply(dirt_)  # 应用脏物体的材质或表面效果
        butil.apply_modifiers(dirt_)  # 应用所有变换

        # 删除远离中心的顶点，留下更小范围内的顶点
        remove_vertices(dirt_, lambda x, y, z: np.sqrt(x**2 + y**2) > radius * 0.92)
        # 调整脏物体的位置，使其略微下降
        dirt_.location[-1] -= 0.02

        # 使用 plant_factory 生成植物对象
        plant = self.plant_factory.spawn_asset(i=i, loc=(0, 0, 0), rot=(0, 0, 0))
        # 将植物的位置调整到最低点
        origin2lowest(plant, approximate=True)
        # 完成植物对象的最终处理
        self.plant_factory.finalize_assets(plant)
        # 根据植物的边界框大小调整其比例
        scale = np.min(
            np.array([self.side_size, self.side_size, self.top_size])
            / np.max(np.abs(np.array(plant.bound_box)), 0)
        )
        plant.scale = [scale] * 3
        # 设置植物的位置，使其与脏物体的 z 坐标对齐
        plant.location[-1] = dirt_z
        # 将基础对象、植物和脏物体合并为一个对象
        obj = join_objects([obj, plant, dirt_])
        return obj


class LargePlantContainerFactory(PlantContainerFactoryOld):
    plant_factories = [MonocotFactory]

    def __init__(self, factory_seed, coarse=False):
        super(LargePlantContainerFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(self.factory_seed):
            self.base_factory.depth = log_uniform(1.0, 1.5)
            self.base_factory.scale = log_uniform(0.15, 0.25)
            self.side_size = (
                self.base_factory.scale * uniform(1.5, 2.0) * self.base_factory.r_expand
            )
            self.top_size = uniform(1, 1.5)
            # if WALL_HEIGHT - 2*WALL_THICKNESS < 3:
            #     self.top_size = uniform(1.5, WALL_HEIGHT - 2*WALL_THICKNESS)
            # else:
            #     self.top_size = uniform(1.5, 3)
            # print(f"{self.side_size=} {self.top_size=} {WALL_THICKNESS=} {WALL_HEIGHT=}")
