# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
from collections import defaultdict, deque
from collections.abc import Sequence

import gin
import networkx as nx
import numpy as np
from numpy.random import uniform

from infinigen.core.constraints.example_solver.room.configs import (
    LOOP_ROOM_TYPES,  # 环状房间类型
    ROOM_CHILDREN,  # 房间的子房间配置
    ROOM_NUMBERS,  # 房间数量约束
    STUDIO_ROOM_CHILDREN,  # 单间公寓的子房间配置
    TYPICAL_AREA_ROOM_TYPES,  # 房间类型的典型面积
    UPSTAIRS_ROOM_CHILDREN,  # 楼上房间的子房间配置
)
from infinigen.core.constraints.example_solver.room.types import (
    RoomGraph,
    RoomType,
    get_room_type,
)
from infinigen.core.constraints.example_solver.room.utils import unit_cast
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform
from infinigen.core.util.random import random_general as rg


@gin.configurable(denylist=["factory_seed", "level"])
class GraphMaker:
    def __init__(
        self,
        factory_seed,  # 随机种子
        level=0,  # 房屋楼层等级（0表示地面层）
        requires_staircase=False,  # 是否需要楼梯
        room_children="home",  # 子房间类型
        typical_area_room_types=TYPICAL_AREA_ROOM_TYPES,  # 房间的典型面积类型
        loop_room_types=LOOP_ROOM_TYPES,
        room_numbers=ROOM_NUMBERS,
        max_cycle_basis=1,
        requires_bathroom_privacy=True,  # 是否需要浴室隐私
        entrance_type=("weighted_choice", (0.5, "porch"), (0.5, "hallway")),
        hallway_alpha=1,
        no_hallway_children_prob=0.4,
    ):
        self.factory_seed = factory_seed
        with FixedSeed(factory_seed):  # 确保生成的图具有固定的随机性
            self.requires_staircase = requires_staircase
            match room_children:  # 根据房间类型选择子房间配置
                case "home":
                    self.room_children = (
                        ROOM_CHILDREN if level == 0 else UPSTAIRS_ROOM_CHILDREN
                    )
                case _:
                    self.room_children = STUDIO_ROOM_CHILDREN
            # 找出所有包含走廊的房间类型
            self.hallway_room_types = [
                r for r, m in self.room_children.items() if RoomType.Hallway in m
            ]
            self.typical_area_room_types = typical_area_room_types
            self.loop_room_types = loop_room_types
            self.room_numbers = room_numbers
            self.max_samples = 1000  # 最大采样次数
            self.slackness = log_uniform(1.5, 1.8)  # 松弛因子，用于面积计算
            self.max_cycle_basis = max_cycle_basis
            self.requires_bathroom_privacy = requires_bathroom_privacy
            self.entrance_type = rg(entrance_type)  # 随机选择入口类型
            self.hallway_prob = lambda x: 1 / (x + hallway_alpha)  # 计算走廊概率
            self.no_hallway_children_prob = no_hallway_children_prob
            self.skewness_min = 0.7

    def make_graph_singleroom(self, i):  # 用于生成房间图的核心方法
        with FixedSeed(i):
            for _ in range(self.max_samples):  # 尝试多次生成以满足约束
                room_type_counts = defaultdict(int)
                rooms = []
                children = defaultdict(list)
                queue = deque()

                def add_room(t, p):
                    i = len(rooms)  # 新房间的索引
                    name = f"{t}_{room_type_counts[t]}"  # 命名格式为 "类型_编号"
                    room_type_counts[t] += 1
                    if p is not None:
                        children[p].append(i)  # 将新房间添加到父房间的子房间列表
                    rooms.append(name)
                    queue.append(i)  # 将新房间加入队列

                # 添加初始房间（如客厅）
                add_room(RoomType.NewRoom, None)
                # while len(queue) > 0:  # 当队列不为空时，不断生成新房间
                #     i = queue.popleft()  # 从队列取出一个房间
                #     # 根据房间类型生成子房间
                #     for rt, spec in self.room_children[get_room_type(rooms[i])].items():
                #         for _ in range(rg(spec)):
                #             # 按照配置生成子房间数量
                #             add_room(rt, i)
                # 检查是否满足浴室隐私要求
                if self.requires_bathroom_privacy and not self.has_bathroom_privacy:
                    continue
                # 添加环状房间关系
                for i, r in enumerate(rooms):
                    for j, s in enumerate(rooms):
                        if (rt := get_room_type(r)) in self.loop_room_types:
                            if (rt_ := get_room_type(s)) in self.loop_room_types[rt]:
                                if (
                                    uniform() < self.loop_room_types[rt][rt_]
                                    and j not in children[i]
                                ):
                                    children[i].append(j)
                # 调整走廊和其他房间的关系
                for i, r in enumerate(rooms):
                    if get_room_type(r) in self.hallway_room_types:
                        hallways = [
                            j
                            for j in children[i]
                            if get_room_type(rooms[j]) == RoomType.Hallway
                        ]
                        other_rooms = [
                            j
                            for j in children[i]
                            if get_room_type(rooms[j]) != RoomType.Hallway
                        ]
                        children[i] = hallways.copy()
                        for k, o in enumerate(other_rooms):
                            if (
                                uniform() < self.no_hallway_children_prob
                                or len(hallways) == 0
                            ):
                                children[i].append(o)
                            else:
                                children[
                                    hallways[np.random.randint(len(hallways))]
                                ].append(o)
                # 处理入口和楼梯房间
                hallways = [
                    i
                    for i, r in enumerate(rooms)
                    if get_room_type(r) == RoomType.Hallway
                ]
                if len(hallways) == 0:
                    entrance = 0  # 如果没有走廊，默认入口为客厅
                else:
                    if self.requires_staircase:
                        prob = np.array(
                            [self.hallway_prob(len(children[h])) for h in hallways]
                        )
                        add_room(
                            RoomType.Staircase,
                            np.random.choice(hallways, p=prob / prob.sum()),
                        )
                    prob = np.array(
                        [self.hallway_prob(len(children[h])) for h in hallways]
                    )
                    entrance = np.random.choice(hallways, p=prob / prob.sum())
                    if self.entrance_type == "porch":
                        add_room(RoomType.Balcony, entrance)
                        entrance = queue.pop()
                    elif self.entrance_type == "none":
                        entrance = None
                # 创建房间图
                children_ = [children[i] for i in range(len(rooms))]
                room_graph = RoomGraph(children_, rooms, entrance)
                if self.satisfies_constraint(room_graph):  # 检查图是否满足约束
                    return room_graph
                
    def make_graph(self, i):  # 用于生成房间图的核心方法
        with FixedSeed(i):
            for _ in range(self.max_samples):  # 尝试多次生成以满足约束
                room_type_counts = defaultdict(int)
                rooms = []
                children = defaultdict(list)
                queue = deque()

                def add_room(t, p):
                    i = len(rooms)  # 新房间的索引
                    name = f"{t}_{room_type_counts[t]}"  # 命名格式为 "类型_编号"
                    room_type_counts[t] += 1
                    if p is not None:
                        children[p].append(i)  # 将新房间添加到父房间的子房间列表
                    rooms.append(name)
                    queue.append(i)  # 将新房间加入队列

                # 添加初始房间（如客厅）
                add_room(RoomType.LivingRoom, None)
                while len(queue) > 0:  # 当队列不为空时，不断生成新房间
                    i = queue.popleft()  # 从队列取出一个房间
                    # 根据房间类型生成子房间
                    for rt, spec in self.room_children[get_room_type(rooms[i])].items():
                        for _ in range(rg(spec)):
                            # 按照配置生成子房间数量
                            add_room(rt, i)
                # 检查是否满足浴室隐私要求
                if self.requires_bathroom_privacy and not self.has_bathroom_privacy:
                    continue
                # 添加环状房间关系
                for i, r in enumerate(rooms):
                    for j, s in enumerate(rooms):
                        if (rt := get_room_type(r)) in self.loop_room_types:
                            if (rt_ := get_room_type(s)) in self.loop_room_types[rt]:
                                if (
                                    uniform() < self.loop_room_types[rt][rt_]
                                    and j not in children[i]
                                ):
                                    children[i].append(j)
                # 调整走廊和其他房间的关系
                for i, r in enumerate(rooms):
                    if get_room_type(r) in self.hallway_room_types:
                        hallways = [
                            j
                            for j in children[i]
                            if get_room_type(rooms[j]) == RoomType.Hallway
                        ]
                        other_rooms = [
                            j
                            for j in children[i]
                            if get_room_type(rooms[j]) != RoomType.Hallway
                        ]
                        children[i] = hallways.copy()
                        for k, o in enumerate(other_rooms):
                            if (
                                uniform() < self.no_hallway_children_prob
                                or len(hallways) == 0
                            ):
                                children[i].append(o)
                            else:
                                children[
                                    hallways[np.random.randint(len(hallways))]
                                ].append(o)
                # 处理入口和楼梯房间
                hallways = [
                    i
                    for i, r in enumerate(rooms)
                    if get_room_type(r) == RoomType.Hallway
                ]
                if len(hallways) == 0:
                    entrance = 0  # 如果没有走廊，默认入口为客厅
                else:
                    if self.requires_staircase:
                        prob = np.array(
                            [self.hallway_prob(len(children[h])) for h in hallways]
                        )
                        add_room(
                            RoomType.Staircase,
                            np.random.choice(hallways, p=prob / prob.sum()),
                        )
                    prob = np.array(
                        [self.hallway_prob(len(children[h])) for h in hallways]
                    )
                    entrance = np.random.choice(hallways, p=prob / prob.sum())
                    if self.entrance_type == "porch":
                        add_room(RoomType.Balcony, entrance)
                        entrance = queue.pop()
                    elif self.entrance_type == "none":
                        entrance = None
                # 创建房间图
                children_ = [children[i] for i in range(len(rooms))]
                room_graph = RoomGraph(children_, rooms, entrance)
                if self.satisfies_constraint(room_graph):  # 检查图是否满足约束
                    return room_graph

    __call__ = make_graph

    def satisfies_constraint(self, graph):
        if not graph.is_planar or len(graph.cycle_basis) > self.max_cycle_basis:
            return False
        for room_type, constraint in self.room_numbers.items():
            if isinstance(constraint, Sequence):
                n_min, n_max = constraint
            else:
                n_min, n_max = constraint, constraint
            if not n_min <= len(graph[room_type]) <= n_max:
                return False
        return True

    def has_bathroom_privacy(self, rooms, children):
        for i, r in rooms:
            if get_room_type(r) == RoomType.LivingRoom:
                has_public_bathroom = any(
                    get_room_type(rooms[j]) == RoomType.Bathroom for j in children[i]
                )
                if not has_public_bathroom:
                    for j in children[i]:
                        if get_room_type(rooms[j] == RoomType.Bedroom):
                            if not any(get_room_type(rooms[k]) for k in children[j]):
                                return False
        return True

    def suggest_dimensions(self, graph, width=None, height=None):
        area = (
            sum([self.typical_area_room_types[get_room_type(r)] for r in graph.rooms])
            * self.slackness
        )
        if width is None and height is None:
            skewness = uniform(self.skewness_min, 1 / self.skewness_min)
            width = unit_cast(np.sqrt(area * skewness).item())
            height = unit_cast(np.sqrt(area / skewness).item())
        elif uniform(0, 1) < 0.5:
            height_ = unit_cast(area / width)
            height = (
                None
                if height_ > height
                and self.skewness_min < height_ / width < 1 / self.skewness_min
                else height_
            )
        else:
            width_ = unit_cast(area / height)
            width = (
                None
                if width_ > width
                and self.skewness_min < width_ / height < 1 / self.skewness_min
                else width_
            )

        return width, height

    def draw(self, graph):
        g = nx.Graph()
        shortnames = [r[:3].upper() + r.split("_")[-1] for r in graph.rooms]
        g.add_nodes_from(shortnames)
        for k in range(len(shortnames)):
            for l in graph.neighbours[k]:
                g.add_edge(shortnames[k], shortnames[l])
        nx.draw_planar(g, with_labels=True)
