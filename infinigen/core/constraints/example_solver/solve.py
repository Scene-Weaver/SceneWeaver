# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

import copy
import importlib
import json
import logging
import math
import os
import re
from pathlib import Path

import bpy
import gin
import numpy as np
from mathutils import Matrix
from tqdm import trange

from infinigen.assets.metascene_assets import GeneralMetaFactory
from infinigen.assets.objaverse_assets import GeneralObjavFactory
from infinigen.assets.threedfront_assets import GeneralThreedFrontFactory
from infinigen.core import tags as t

# from debug import invisible_others, visible_others
from infinigen.core.constraints import constraint_language as cl
from infinigen.core.constraints import reasoning as r
from infinigen.core.constraints import usage_lookup
from infinigen.core.constraints.constraint_language import util as iu
from infinigen.core.constraints.evaluator import domain_contains
from infinigen.core.constraints.example_solver import (
    greedy,
    propose_continous,
    propose_discrete,
)
from infinigen.core.constraints.example_solver.geometry.dof import (
    apply_relations_surfacesample,
)
from infinigen.core.constraints.example_solver.propose_discrete import moves
from infinigen.core.constraints.example_solver.state_def import State
from infinigen.core.tags import Semantics
from infinigen.core.util import blender as butil
from infinigen_examples.util import constraint_util as cu

from . import moves, propose_relations, state_def
from .annealing import SimulatedAnnealingSolver
from .room import MultistoryRoomSolver, RoomSolver

# from infinigen_examples.steps.tools import calc_position_bias


logger = logging.getLogger(__name__)

GLOBAL_GENERATOR_SINGLETON_CACHE = {}


def map_range(x, xmin, xmax, ymin, ymax, exp=1):
    if x < xmin:
        return ymin
    if x > xmax:
        return ymax

    t = (x - xmin) / (xmax - xmin)
    return ymin + (ymax - ymin) * t**exp


@gin.register
class LinearDecaySchedule:
    def __init__(self, start, end, pct_duration):
        self.start = start
        self.end = end
        self.pct_duration = pct_duration

    def __call__(self, t):
        return map_range(t, 0, self.pct_duration, self.start, self.end)


@gin.configurable
class Solver:
    def __init__(
        self,
        output_folder: Path,
        multistory: bool = False,
        restrict_moves: list = None,
        addition_weight_scalar: float = 1.0,
    ):
        """Initialize the solver

        Parameters
        ----------
        output_folder : Path
            The folder to save output plots to
        print_report_freq : int
            How often to print loss reports
        multistory : bool
            Whether to use the multistory room solver
        constraints_greedy_unsatisfied : str | None
            What do we do if relevant constraints are unsatisfied at the end of a greedy stage?
            Options are 'warn` or `abort` or None

        """

        self.output_folder = output_folder

        self.optim = SimulatedAnnealingSolver(
            output_folder=output_folder,
        )

        self.room_solver_fn = MultistoryRoomSolver if multistory else RoomSolver
        self.state: State = None
        self.all_roomtypes = None
        self.dimensions = None

        self.moves = self._configure_move_weights(
            restrict_moves, addition_weight_scalar=addition_weight_scalar
        )

    def _configure_move_weights(self, restrict_moves, addition_weight_scalar=1.0):
        schedules = {
            "addition": (
                propose_discrete.propose_addition,
                LinearDecaySchedule(
                    6 * addition_weight_scalar, 0.1 * addition_weight_scalar, 0.9
                ),
            ),
            "deletion": (
                propose_discrete.propose_deletion,
                LinearDecaySchedule(2, 0.0, 0.5),
            ),
            "plane_change": (
                propose_discrete.propose_relation_plane_change,
                LinearDecaySchedule(2, 0.1, 1),
            ),
            "resample_asset": (
                propose_discrete.propose_resample,
                LinearDecaySchedule(1, 0.1, 0.7),
            ),
            "reinit_pose": (
                propose_continous.propose_reinit_pose,
                LinearDecaySchedule(1, 0.5, 1),
            ),
            "translate": (propose_continous.propose_translate, 1),
            "rotate": (propose_continous.propose_rotate, 0.5),
        }

        if restrict_moves is not None:
            schedules = {k: v for k, v in schedules.items() if k in restrict_moves}
            logger.info(
                f"Restricting {self.__class__.__name__} moves to {list(schedules.keys())}"
            )

        return schedules

    @gin.configurable
    def choose_move_type(
        self,
        it: int,
        max_it: int,
    ):
        t = it / max_it
        names, confs = zip(*self.moves.items())
        funcs, scheds = zip(*confs)
        weights = np.array([s if isinstance(s, (float, int)) else s(t) for s in scheds])
        return np.random.choice(funcs, p=weights / weights.sum())

    def solve_rooms(self, scene_seed, consgraph: cl.Problem, filter: r.Domain):
        self.state, self.all_roomtypes, self.dimensions = self.room_solver_fn(
            scene_seed
        ).solve()
        return self.state

    @gin.configurable
    def solve_objects(
        self,
        consgraph: cl.Problem,
        filter_domain: r.Domain,
        var_assignments: dict[str, str],
        n_steps: int,
        desc: str,
        abort_unsatisfied: bool = False,
        print_bounds: bool = False,
        expand_collision: bool = False,
        use_initial=False,
    ):
        filter_domain = copy.deepcopy(filter_domain)
        """
        Domain({Semantics.Object, -Semantics.Room}, [
            (StableAgainst({}, {Subpart.SupportSurface, Subpart.Visible, -Subpart.Ceiling, -Subpart.Wall}), Domain({Semantics.Bathroom, Variable(room), Semantics.Room, -Semantics.Object}, [])),
            (-AnyRelation(), Domain({Semantics.Object, -Semantics.Room}, []))
        ])
        """
        desc_full = (desc, *var_assignments.values())

        dom_assignments = {
            k: r.Domain(self.state.objs[objkey].tags)
            for k, objkey in var_assignments.items()
        }

        if use_initial:
            dom_assignments[cu.variable_obj] = r.Domain(
                {Semantics.Object, -Semantics.Room}
            )

        filter_domain = r.substitute_all(filter_domain, dom_assignments)
        """
        Domain({Semantics.Object, -Semantics.Room}, [
            (StableAgainst({}, {Subpart.SupportSurface, Subpart.Visible, -Subpart.Ceiling, -Subpart.Wall}), Domain({SpecificObject(name='bathroom_0-0'), Semantics.Room, Semantics.Bathroom, -Semantics.Object}, [])),
            (-AnyRelation(), Domain({Semantics.Object, -Semantics.Room}, []))
        ])
        """

        if not r.domain_finalized(filter_domain):
            raise ValueError(
                f"Cannot solve {desc_full=} with non-finalized domain {filter_domain}"
            )

        orig_bounds = r.constraint_bounds(consgraph)  # len(orig_bounds) = 63
        # find objects than can be add to fit requirment
        print_bounds = True
        bounds = propose_discrete.preproc_bounds(
            orig_bounds, self.state, filter_domain, print_bounds=print_bounds
        )

        active_count = greedy.update_active_flags(self.state, var_assignments)  # 5,17

        n_start = len(self.state.objs)  # 37
        logger.info(
            f"Greedily solve {desc_full} - stage has {len(bounds)}/{len(orig_bounds)} bounds, "
            f"{active_count=}/{len(self.state.objs)} objs"
        )

        self.optim.reset(max_iters=n_steps)

        ra = (
            trange(n_steps) if self.optim.print_report_freq == 0 else range(n_steps)
        )  # *len(self.state.objs))

        # loc_record = dict()
        # for objname in self.state.objs.keys():
        #     if objname.startswith("window") or objname.startswith("entrance") or objname.startswith("newroom_0-0"):
        #         continue
        #     obj_state = self.state.objs[objname]
        #     loc_record[objname] = obj_state.obj.location.copy()
        #     # np.allclose(x1, x2):

        # 进行迭代
        for j in ra:
            print(j)

            move_gen = propose_continous.propose_translate_all  # 选择移动类型
            relplane_gen = (
                propose_discrete.propose_relation_plane_change_all
            )  # 选择移动类型
            import random

            gen = random.choice([move_gen])

            self.optim.step(
                consgraph, self.state, gen, filter_domain, expand_collision
            )  # MARK # 执行优化步骤

            # Finish = True
            # for objname in self.state.objs.keys():
            #     if objname.startswith("window") or objname.startswith("entrance") or objname.startswith("newroom_0-0"):
            #         continue
            #     obj_state = self.state.objs[objname]
            #     loc = obj_state.obj.location.copy()
            #     loc_old = loc_record[objname]
            #     # print(objname,loc-loc_old)
            #     if np.allclose(loc_old[0], loc[0]) and \
            #         np.allclose(loc_old[1], loc[1]) and \
            #         np.allclose(loc_old[2], loc[2]):
            #         continue
            #     else:
            #         Finish = False
            #         break

            # if j>150 and Finish:
            #     break

        self.optim.save_stats(
            self.output_folder / f"optim_{desc}.csv"
        )  # 保存优化统计信息

        logger.info(
            f"Finished solving {desc_full}, added {len(self.state.objs) - n_start} "
            f"objects, loss={self.optim.curr_result.loss():.4f} viol={self.optim.curr_result.viol_count()}"
        )

        logger.info(self.optim.curr_result.to_df())

        violations = {
            k: v for k, v in self.optim.curr_result.violations.items() if v > 0
        }

        if len(violations):
            msg = f"Solver has failed to satisfy constraints for stage {desc_full}. {violations=}."
            if abort_unsatisfied:
                butil.save_blend(self.output_folder / f"abort_{desc}.blend")
                raise ValueError(msg)
            else:
                msg += " Continuing anyway, override `solve_objects.abort_unsatisfied=True` via gin to crash instead."
                logger.warning(msg)

        # re-enable everything so the blender scene populates / displays correctly etc
        for k, v in self.state.objs.items():
            greedy.set_active(self.state, k, True)

        return self.state

    @gin.configurable
    def add_rule(
        self,
        consgraph: cl.Problem,
        filter_domain: r.Domain,
        var_assignments: dict[str, str],
        n_steps: int,
        desc: str,
        abort_unsatisfied: bool = False,
        print_bounds: bool = False,
        expand_collision: bool = False,
        use_initial=False,
    ):
        filter_domain = copy.deepcopy(filter_domain)
        """
        Domain({Semantics.Object, -Semantics.Room}, [
            (StableAgainst({}, {Subpart.SupportSurface, Subpart.Visible, -Subpart.Ceiling, -Subpart.Wall}), Domain({Semantics.Bathroom, Variable(room), Semantics.Room, -Semantics.Object}, [])),
            (-AnyRelation(), Domain({Semantics.Object, -Semantics.Room}, []))
        ])
        """
        desc_full = (desc, *var_assignments.values())

        dom_assignments = {
            k: r.Domain(self.state.objs[objkey].tags)
            for k, objkey in var_assignments.items()
        }

        if use_initial:
            dom_assignments[cu.variable_obj] = r.Domain(
                {Semantics.Object, -Semantics.Room}
            )

        filter_domain = r.substitute_all(filter_domain, dom_assignments)

        if not r.domain_finalized(filter_domain):
            raise ValueError(
                f"Cannot solve {desc_full=} with non-finalized domain {filter_domain}"
            )

        orig_bounds = r.constraint_bounds(consgraph)
        # find objects than can be add to fit requirment

        print_bounds = True
        bounds = propose_discrete.preproc_bounds(
            orig_bounds, self.state, filter_domain, print_bounds=print_bounds
        )

        active_count = greedy.update_active_flags(self.state, var_assignments)

        n_start = len(self.state.objs)  # 37
        logger.info(
            f"Greedily solve {desc_full} - stage has {len(bounds)}/{len(orig_bounds)} bounds, "
            f"{active_count=}/{len(self.state.objs)} objs"
        )

        self.optim.reset(max_iters=n_steps)

        ra = (
            trange(n_steps) if self.optim.print_report_freq == 0 else range(n_steps)
        )  # *len(self.state.objs))

        # 进行迭代
        for j in ra:
            print(j)

            # move_gen = propose_discrete.propose_addition
            move_gen = self.choose_move_type_candid(j, n_steps)  # 选择移动类型

            self.optim.step(
                consgraph, self.state, move_gen, filter_domain, expand_collision
            )  # MARK # 执行优化步骤

        self.optim.save_stats(
            self.output_folder / f"optim_{desc}.csv"
        )  # 保存优化统计信息

        logger.info(
            f"Finished solving {desc_full}, added {len(self.state.objs) - n_start} "
            f"objects, loss={self.optim.curr_result.loss():.4f} viol={self.optim.curr_result.viol_count()}"
        )

        logger.info(self.optim.curr_result.to_df())

        violations = {
            k: v for k, v in self.optim.curr_result.violations.items() if v > 0
        }

        if len(violations):
            msg = f"Solver has failed to satisfy constraints for stage {desc_full}. {violations=}."
            if abort_unsatisfied:
                butil.save_blend(self.output_folder / f"abort_{desc}.blend")
                raise ValueError(msg)
            else:
                msg += " Continuing anyway, override `solve_objects.abort_unsatisfied=True` via gin to crash instead."
                logger.warning(msg)

        # re-enable everything so the blender scene populates / displays correctly etc
        for k, v in self.state.objs.items():
            greedy.set_active(self.state, k, True)

        return self.state

    def load_gpt_results(self):
        json_name = os.getenv("JSON_RESULTS")
        with open(json_name, "r") as f:
            info = json.load(f)
        self.name_mapping = info["name_mapping"]
        if "Placement_big" in info:
            self.Placement_big = info["Placement_big"]
            if "Placement_small" in info:
                self.Placement_small = info["Placement_small"]
        else:
            self.Placement = info["Placement"]
        self.category_against_wall = info["category_against_wall"]
        self.big_category_dict = info["big_category_dict"]
        self.retrieve_objav_assets(self.big_category_dict, self.name_mapping)
        return

    def retrieve_objav_assets(self, category_cnt, name_mapping=None):
        save_dir = os.getenv("save_dir")
        # if os.path.exists(f"{save_dir}/objav_files.json"):
        #     with open(f"{save_dir}/objav_files.json", "r") as f:
        #         self.LoadObjavFiles = json.load(f)
        #     return

        def get_case_insensitive(dictionary, key):
            return next(
                (v for k, v in dictionary.items() if k.lower() == key.lower()), None
            )

        # retrieve objaverse
        self.LoadObjavCnts = dict()
        for name in category_cnt.keys():
            if name_mapping is not None and name not in name_mapping:
                name = name.lower()
            if name_mapping is None or name_mapping[name] is None:
                self.LoadObjavCnts[name] = get_case_insensitive(category_cnt, name)

        with open(f"{save_dir}/objav_cnts.json", "w") as f:
            json.dump(self.LoadObjavCnts, f, indent=4)

        # cmd = """
        # source /home/yandan/anaconda3/etc/profile.d/conda.sh
        # conda activate idesign
        # python ~/workspace/SceneWeaver/infinigen/assets/objaverse_assets/retrieve_idesign.py > run.log 2>&1
        # """
        # subprocess.run(["bash", "-c", cmd])
       
        os.system(
            f'env -i bash --norc --noprofile -c "./run/retrieve.sh {save_dir}" > run.log 2>&1'
        )
        with open(f"{save_dir}/objav_files.json", "r") as f:
            self.LoadObjavFiles = json.load(f)
        return

    @gin.configurable
    def add_graph_gpt(
        self,
        # filter_domain: r.Domain,
        iter,
        var_assignments: dict[str, str],
        stage="large",  # large, medium, small
    ):
        self.del_no_relation_objects()
        json_name = os.getenv("JSON_RESULTS")
        if json_name == "Nothing":
            return self.state
        with open(json_name, "r") as f:
            info = json.load(f)

        self.Placement = info["Placement"]
        self.category_against_wall = info["category_against_wall"]
        if "category_on_the_floor" in info:
            self.category_on_the_floor = info["category_on_the_floor"]
        else:
            self.category_on_the_floor = None
        self.name_mapping = dict()
        for k, v in info["name_mapping"].items():
            self.name_mapping[k.lower()] = v
        # after loading name mapping, retrieve objects.
        self.retrieve_objav_assets(info["Number of new furniture"], self.name_mapping)

        ordered_names = self.get_ordered_objects(self.Placement)
        # for key, value in Placement.items():
        for step in ["large", "medium", "small"]:
            for key in ordered_names:
                value = self.Placement[key]
                for num in value.keys():
                    # for step in ["large", "medium", "small"]:
                    #     for key, value in self.Placement.items():
                    #         for num in value.keys():
                    if not num.isdigit():
                        print(f"🎯 Error adding object {key}")
                        continue

                    # remove relation with room,focus on relation with objects
                    if (
                        "parent" in value[num]
                        and value[num]["parent"] != []
                        and isinstance(value[num]["parent"][0], list)
                    ):
                        value[num]["parent"] = [
                            i for i in value[num]["parent"] if i[0] != "newroom_0-0"
                        ]
                        if len(value[num]["parent"]) > 0:
                            value[num]["parent"] = value[num]["parent"][
                                0
                            ]  # can only have one relation with objects
                    asset_file = None
                    position = value[num]["position"]
                    if len(value[num]["position"]) == 2:
                        position += [0]

                    rotation = value[num]["rotation"]
                    if isinstance(rotation, list):
                        rotation = rotation[2]
                    elif isinstance(rotation, int):
                        rotation = rotation * math.pi / 180
                    else:
                        AssertionError
                    size = value[num]["size"]
                    name = key.lower()

                    if name.lower() not in self.name_mapping:
                        print(f"Error: objects {name} has not mapping !! ")
                        continue
                    module_and_class = self.name_mapping[name.lower()]
                    if (
                        "parent" in value[num]
                        and value[num]["parent"] != []
                        and value[num]["parent"][0] != "newroom_0-0"
                    ):
                        if value[num]["parent"][1] in ["on", "ontop"] or (
                            len(value[num]["parent"]) == 3
                            and value[num]["parent"][2] in ["on", "ontop"]
                        ):
                            stage = "small"
                            if stage != step:
                                continue
                            try:
                                parent_obj_name, relation = value[num]["parent"]
                            except:
                                parent_key, parent_num, relation = value[num]["parent"]
                                if "name" not in self.Placement[parent_key][parent_num]:
                                    continue
                                parent_obj_name = self.Placement[parent_key][
                                    parent_num
                                ]["name"]

                            against_wall = False
                            on_floor = False
                            size = [-1, -1, -1]
                        else:
                            stage = "medium"
                            if stage != step:
                                continue
                            try:
                                parent_obj_name, relation = value[num]["parent"]
                            except:
                                parent_key, parent_num, relation = value[num]["parent"]
                                parent_obj_name = self.Placement[parent_key][
                                    parent_num
                                ]["name"]
                            against_wall = (
                                True if key in self.category_against_wall else False
                            )
                            if self.category_on_the_floor is not None:
                                on_floor = (
                                    True if key in self.category_on_the_floor else False
                                )
                            else:
                                on_floor = True

                    else:
                        stage = "large"
                        if stage != step:
                            continue
                        parent_obj_name = None

                        against_wall = (
                            True if key in self.category_against_wall else False
                        )
                        if self.category_on_the_floor is not None:
                            on_floor = (
                                True if key in self.category_on_the_floor else False
                            )
                        else:
                            on_floor = True
                        if not on_floor and not against_wall:  # floating object
                            continue

                    filter_domain = self.calc_filter_domain(
                        value, num, on_floor=on_floor, against_wall=against_wall
                    )

                    if module_and_class is None:
                        gen_class = GeneralObjavFactory
                        size = value[num]["size"]
                        x_dim, y_dim, z_dim = size
                        category = name
                        gen_class._x_dim = x_dim
                        gen_class._y_dim = y_dim
                        gen_class._z_dim = z_dim
                        gen_class._category = category

                        class_name = category
                        asset_file = self.LoadObjavFiles[category][0]
                    else:
                        module_name, class_name = module_and_class.rsplit(".", 1)
                        module = importlib.import_module(
                            "infinigen.assets.objects." + module_name
                        )
                        class_obj = getattr(module, class_name)
                        gen_class = class_obj
                    search_rels = filter_domain.relations
                    # 筛选出有效的关系，只选择非否定关系
                    search_rels = [
                        rd
                        for rd in search_rels
                        if not isinstance(rd[0], cl.NegatedRelation)
                    ]

                    assign = propose_relations.find_given_assignments(
                        self.state, search_rels, parent_obj_name=parent_obj_name
                    )
                    for i, assignments in enumerate(assign):
                        found_tags = usage_lookup.usages_of_factory(gen_class)
                        move = moves.Addition(
                            names=[
                                f"{np.random.randint(1e6):04d}_{gen_class.__name__}"
                            ],  # decided later # 随机生成一个名称，基于生成器类的名称
                            gen_class=gen_class,  # 使用传入的生成器类
                            relation_assignments=assignments,  # 传入分配的关系
                            temp_force_tags=found_tags,  # 临时强制标签
                        )

                        while True:
                            target_name = f"{np.random.randint(1e7)}_{class_name}"
                            if target_name not in self.state.objs:
                                break
                        # target_name = np.random.randint(1e7)+"_SofaFactory"

                        move.apply_init(
                            self.state,
                            target_name,
                            size,
                            position,
                            rotation,
                            gen_class,
                            asset_file=asset_file,
                        )

                        self.Placement[key][num]["name"] = target_name

                        break
                    # invisible_others()
                    # bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
                    # visible_others()

        return self.state

    @gin.configurable
    def add_obj_crowd(
        self,
        # filter_domain: r.Domain,
        iter,
        var_assignments: dict[str, str],
    ):
        self.del_no_relation_objects()

        #{
        #     "User demand": "BookStore",
        #     "Roomsize": [3, 4],
        #     "Relation": "on",
        #     "Parent ID": "2245622_LargeShelfFactory"
        #     "Number of new furniture": {"book":"30", "frame":"5", "vase":3},
        # }
        json_name = os.getenv("JSON_RESULTS")
        if json_name == "Nothing":
            return self.state
        with open(json_name, "r") as f:
            info = json.load(f)

        parent_obj_name = info["Parent ID"]
        Placement = info["Number of new furniture"]
        relation = info["Relation"]

        self.category_against_wall = None
        self.category_on_the_floor = None
        self.name_mapping = dict()
        for k, v in info["name_mapping"].items():
            self.name_mapping[k.lower()] = v


        for name,cnt in Placement.items():
            for i in range(int(cnt)):
                category = name
                module_and_class = self.name_mapping[name.lower()]
                against_wall = False
                on_floor = False
                

                filter_domain = self.calc_filter_domain(
                    category,
                    num=None,
                    on_floor=on_floor,
                    against_wall=against_wall,
                    parent_obj_name=parent_obj_name,
                    relation=relation,
                )
                import random

                if module_and_class is None:
                    continue  # TODO, only support infinigen objects in this function right now, since no size info is provided for other assets.

                module_name, class_name = module_and_class.rsplit(".", 1)
                module = importlib.import_module("infinigen.assets.objects." + module_name)
                class_obj = getattr(module, class_name)
                gen_class = class_obj

                # gen_class = GeneralObjavFactory
                # size = [  ]
                search_rels = filter_domain.relations
                # 筛选出有效的关系，只选择非否定关系
                search_rels = [
                    rd for rd in search_rels if not isinstance(rd[0], cl.NegatedRelation)
                ]

                assign = propose_relations.find_given_assignments(
                    self.state, search_rels, parent_obj_name=parent_obj_name
                )
                for i, assignments in enumerate(assign):
                    found_tags = usage_lookup.usages_of_factory(gen_class)
                    move = moves.Addition(
                        names=[
                            f"{np.random.randint(1e6):04d}_{gen_class.__name__}"
                        ],  # decided later # 随机生成一个名称，基于生成器类的名称
                        gen_class=gen_class,  # 使用传入的生成器类
                        relation_assignments=assignments,  # 传入分配的关系
                        temp_force_tags=found_tags,  # 临时强制标签
                    )

                    while True:
                        target_name = f"{np.random.randint(1e7)}_{class_name}"
                        if target_name not in self.state.objs:
                            break

                    success = move.apply_random(
                        self.state,
                        target_name,
                        gen_class,
                    )

                    # if not success:
                    #     bpy.data.objects.remove(self.state.objs[target_name].obj)
                    #     del self.state.objs[target_name]
                    break

        return self.state

    def add_relation(self):
        layouts = dict()
        import os

        json_name = os.getenv("JSON_RESULTS")
        with open(json_name, "r") as f:
            j = json.load(f)
        for key, value in j.items():
            layouts[key] = value

        for name, info in layouts.items():
            if "keyboard" in name:
                a = 1
            print("adding relation for ", name)
            os = self.state.objs[name]
            self.add_relation_obj(name, info["parent"])

        return

    def add_relation_obj(self, child_name, new_relations):
        from infinigen_examples.steps.tools import export_relation

        objinfo = self.state.objs[child_name]
        for new_rel in new_relations:
            obj_relations = [
                [rel.target_name, export_relation(rel.relation)]
                for rel in objinfo.relations
                if rel.target_name != "newroom_0-0"
            ]
            room_relations = [
                [rel.target_name, export_relation(rel.relation)]
                for rel in objinfo.relations
                if rel.target_name == "newroom_0-0"
            ]
            old_relations = [
                [rel.target_name, export_relation(rel.relation)]
                for rel in objinfo.relations
            ]

            if new_rel in old_relations:
                continue

            parent_name, rel_name = new_rel
            # relation number is limited
            if parent_name != "newroom_0-0" and len(obj_relations) >= 1:
                continue
            elif parent_name == "newroom_0-0" and len(room_relations) >= 2:
                continue
            else:
                print("adding relation: ", child_name, parent_name, rel_name)
                self.add_new_relation(child_name, parent_name, rel_name)

        return

    def add_new_relation(self, child_name, parent_name, relation):
        all_room = r.Domain({t.Semantics.Room, -t.Semantics.Object})
        all_obj = r.Domain({t.Semantics.Object, -t.Semantics.Room})

        if relation == "against_wall":
            base_domain = all_obj.with_relation(cu.against_wall, all_room)
        elif relation == "side_against_wall":
            base_domain = all_obj.with_relation(cu.side_against_wall, all_room)
        elif relation == "on_floor":
            base_domain = all_obj.with_relation(cu.on_floor, all_room)
        else:
            if parent_name == "newroom_0-0":
                return
            module_name = self.state.objs[parent_name].generator.__module__
            attribute_name = self.state.objs[parent_name].generator.__class__.__name__
            module = importlib.import_module(module_name)
            parent_Factory = getattr(module, attribute_name)
            parent_domain = r.Domain(usage_lookup.usages_of_factory(parent_Factory))
            relation_module = getattr(cu, relation)
            base_domain = all_obj.with_relation(relation_module, parent_domain)

        rel = base_domain.relations[-1][0]
        assignment = state_def.RelationState(
            relation=rel,  # 当前关系
            target_name=parent_name,  # 目标对象
            child_plane_idx=0,  # TODO fill in at apply()-time
            parent_plane_idx=0,  # 当前父对象的平面索引
        )
        # check if relation has already been added
        for rel in self.state.objs[child_name].relations:
            if (
                rel.target_name == assignment.target_name
                and rel.relation.child_tags == assignment.relation.child_tags
                and rel.relation.parent_tags == assignment.relation.parent_tags
            ):
                return

        self.state.objs[child_name].relations.append(assignment)

        parent_planes = apply_relations_surfacesample(
            self.state, child_name, use_initial=True, closest_surface=True
        )
        return

    # def add_against_wall(self,target_name):
    #     all_room = r.Domain({t.Semantics.Room, -t.Semantics.Object})
    #     all_obj = r.Domain({t.Semantics.Object, -t.Semantics.Room})
    #     base_domain = all_obj.with_relation(cu.against_wall, all_room)
    #     rel = base_domain.relations[-1][0]
    #     # parent_obj = obj = self.state.objs['newroom_0-0'].obj
    #     # n_parent_planes = len(
    #     #         self.state.planes.get_tagged_planes(parent_obj, rel.parent_tags)
    #     #     )
    #     # parent_order = np.arange(n_parent_planes)
    #     # np.random.shuffle(parent_order)

    #     assignment = state_def.RelationState(
    #             relation=rel,  # 当前关系
    #             target_name='newroom_0-0',  # 目标对象
    #             child_plane_idx=0,  # TODO fill in at apply()-time
    #             parent_plane_idx=0,  # 当前父对象的平面索引
    #         )

    #     for rel in self.state.objs[target_name].relations:
    #         if rel.target_name == assignment.target_name \
    #             and rel.relation.child_tags == assignment.relation.child_tags \
    #             and rel.relation.parent_tags == assignment.relation.parent_tags:
    #             return

    #     self.state.objs[target_name].relations.append(assignment)
    #     return

    def update_graph(self):
        from infinigen_examples.steps.tools import calc_position_bias

        layouts = dict()
        import os

        json_name = os.getenv("JSON_RESULTS")
        with open(json_name, "r") as f:
            j = json.load(f)
        for key, value in j.items():
            layouts[key] = value


        for name, info in layouts.items():
            if name not in self.state.objs:
                print(
                    f"Error: object {name} is not in the current scene, obmit this object !!"
                )
                continue
            os = self.state.objs[name]
            obj = os.obj

            obj.rotation_mode = "XYZ"
            iu.set_rotation(self.state.trimesh_scene, os.obj.name, info["rotation"])
            # Force update
            bpy.context.view_layer.update()
            offset_vector = np.array(calc_position_bias(obj))
            new_loc = np.array(info["location"]) - offset_vector
            iu.set_location(self.state.trimesh_scene, os.obj.name, new_loc)

            spawn_asset = bpy.data.objects[os.populate_obj]
            spawn_asset.location = new_loc
            spawn_asset.rotation_euler = info["rotation"]

            size = info["size"]

            scale_x = size[0] / obj.dimensions[0] if obj.dimensions[0] != 0 else 1
            scale_y = size[1] / obj.dimensions[1] if obj.dimensions[1] != 0 else 1
            scale_z = (
                max(size[2], 0.01) / obj.dimensions[2] if obj.dimensions[2] != 0 else 1
            )
            obj.scale = (scale_x, scale_y, scale_z)
            bpy.context.view_layer.objects.active = obj  # Set as active object
            obj.select_set(True)  # Select the object
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

            parent_planes = apply_relations_surfacesample(
                self.state, name, use_initial=True, closest_surface=True
            )

        remove_lst = []
        for name in self.state.objs:
            if name not in layouts:
                remove_lst.append(name)

        for name in remove_lst:
            if name.startswith("window") or name == "newroom_0-0" or name == "entrance":
                continue
            self.delete_object(name)
            # spawn_asset_name = self.state.objs[name].populate_obj
            # spawn_obj = bpy.data.objects.get(spawn_asset_name)
            # # bpy.data.objects.remove(spawn_obj)
            # delete_object_with_children(spawn_obj)
            # placeholder_obj = self.state.objs[name].obj
            # # bpy.data.objects.remove(placeholder_obj)
            # delete_object_with_children(placeholder_obj)

            # self.state.objs.pop(name)

        self.del_no_relation_objects()

        return

    def update_graph_size(self):
        from infinigen_examples.steps.tools import calc_position_bias

        layouts = dict()
        import os

        json_name = os.getenv("JSON_RESULTS")
        with open(json_name, "r") as f:
            j = json.load(f)
        for key, value in j.items():
            layouts[key] = value

        for name, info in layouts.items():
            if name not in self.state.objs:
                print(
                    f"Error: object {name} is not in the current scene, obmit this object !!"
                )
                continue
            os = self.state.objs[name]
            obj = os.obj

            obj.rotation_mode = "XYZ"
            iu.set_rotation(self.state.trimesh_scene, os.obj.name, info["rotation"])
            # Force update
            bpy.context.view_layer.update()
            offset_vector = np.array(calc_position_bias(obj))
            new_loc = np.array(info["location"]) - offset_vector
            iu.set_location(self.state.trimesh_scene, os.obj.name, new_loc)

            spawn_asset = bpy.data.objects[os.populate_obj]
            spawn_asset.location = new_loc
            spawn_asset.rotation_euler = info["rotation"]

            size = info["size"]

            scale_x = size[0] / obj.dimensions[0] if obj.dimensions[0] != 0 else 1
            scale_y = size[1] / obj.dimensions[1] if obj.dimensions[1] != 0 else 1
            scale_z = size[2] / obj.dimensions[2] if obj.dimensions[2] != 0 else 1
            obj.scale = (scale_x, scale_y, scale_z)
            bpy.context.view_layer.objects.active = obj  # Set as active object
            obj.select_set(True)  # Select the object
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

            parent_planes = apply_relations_surfacesample(
                self.state, name, use_initial=True, closest_surface=True
            )

        return

    def del_no_relation_objects(self):
        for name in list(self.state.objs.keys())[::-1]:
            relations = self.state.objs[name].relations
            for rel in relations[::-1]:
                if rel.target_name not in self.state.objs.keys():
                    relations.remove(rel)
            if relations == [] and self.state.objs[name].generator is not None:
                # delete_object_with_children(self.state.objs[name].obj)
                self.delete_object(name)
                # self.state.objs.pop(name)

        return

    def delete_object(self, name):
        if name in self.state.objs:
            objname = self.state.objs[name].obj.name
            from infinigen.core.constraints.constraint_language.util import (
                delete_obj_with_children,
            )

            delete_obj_with_children(
                self.state.trimesh_scene,
                objname,
                delete_blender=True,
                delete_asset=True,
            )
            self.state.objs.pop(name)
            print(f"!!! Deleting object {name} and its children")
        return

    def remove_object(self):
        json_name = os.getenv("JSON_RESULTS")
        with open(json_name, "r") as f:
            j = json.load(f)
            remove_lst = j["objects to remove"]

        for name in remove_lst:
            if name.startswith("window") or name == "newroom_0-0" or name == "entrance":
                continue
            self.delete_object(name)
            # spawn_asset_name = self.state.objs[name].populate_obj
            # spawn_obj = bpy.data.objects.get(spawn_asset_name)
            # # bpy.data.objects.remove(spawn_obj)
            # delete_object_with_children(spawn_obj)
            # placeholder_obj = self.state.objs[name].obj
            # # bpy.data.objects.remove(placeholder_obj)
            # delete_object_with_children(placeholder_obj)
            # self.state.objs.pop(name)

        self.del_no_relation_objects()

        return

    def get_ordered_objects(self, placement_dict):
        # Collect all object types
        object_types = list(placement_dict.keys())

        # Build dependency graph and in-degree map
        graph = {ot: [] for ot in object_types}
        in_degree = {ot: 0 for ot in object_types}

        for obj_type in object_types:
            instances = placement_dict[obj_type]
            for instance_id, instance in instances.items():
                if (
                    "parent" in instance
                    and instance["parent"] is not None
                    and len(instance["parent"]) > 0
                ):
                    if isinstance(instance["parent"][0], str):
                        parent_type = instance["parent"][0]
                    elif isinstance(instance["parent"][0], list):
                        parent_type = instance["parent"][0][0]
                    else:
                        AssertionError("order error")
                    if parent_type in graph:
                        graph[parent_type].append(obj_type)
                        in_degree[obj_type] += 1

        # Kahn's algorithm for topological sort
        queue = [ot for ot in object_types if in_degree[ot] == 0]
        ordered = []

        while queue:
            node = queue.pop(0)
            ordered.append(node)
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Check for cycles (unlikely here)
        if len(ordered) != len(object_types):
            # import pdb
            # pdb.set_trace()
            # raise ValueError("Cycle detected in dependencies")
            ordered += [i for i in object_types if i not in ordered]

        return ordered

    @gin.configurable
    def init_graph_gpt(
        self,
        var_assignments: dict[str, str],
    ):
        Placement = self.Placement_big
        ordered_names = self.get_ordered_objects(Placement)
        # for key, value in Placement.items():
        for stage in ["large", "medium"]:
            for key in ordered_names:
                value = Placement[key]
                if key == "coffeeTable":
                    a = 1
                for num in value.keys():
                    position = value[num]["position"]
                    if len(value[num]["position"]) == 2:
                        position += [0.14]
                    rotation = value[num]["rotation"] * math.pi / 180
                    size = value[num]["size"]
                    name = key
                    if name not in self.name_mapping:
                        name = name.lower()
                    if name not in self.name_mapping:
                        continue
                    module_and_class = self.name_mapping[name]
                    if stage == "small":
                        pass
                    #     this_stage = "small"
                    #     parent_key, parent_num, relation = value[num]["parent"]
                    #     parent_obj_name = self.Placement_big[parent_key][parent_num]["name"]
                    #     against_wall = False
                    #     on_floor = False
                    #     size = [-1, -1, -1]
                    else:
                        if (
                            "parent" in value[num]
                            and value[num]["parent"] is not None
                            and value[num]["parent"] != []
                        ):
                            this_stage = "medium"
                            if this_stage != stage:
                                continue
                            parent_key, parent_num, relation = value[num]["parent"]
                            try:
                                parent_obj_name = self.Placement_big[parent_key][
                                    parent_num
                                ]["name"]
                            except:
                                parent_obj_name = None
                        else:
                            this_stage = "large"
                            if this_stage != stage:
                                continue
                            parent_obj_name = None

                        against_wall = (
                            True if key in self.category_against_wall else False
                        )
                        on_floor = True

                    filter_domain = self.calc_filter_domain(
                        value, num, on_floor=on_floor, against_wall=against_wall
                    )

                    if module_and_class is None:
                        gen_class = GeneralObjavFactory
                        size = value[num]["size"]
                        x_dim, y_dim, z_dim = size
                        category = name
                        gen_class._x_dim = x_dim
                        gen_class._y_dim = y_dim
                        gen_class._z_dim = z_dim
                        gen_class._category = category

                        class_name = category
                    else:
                        module_name, class_name = module_and_class.rsplit(".", 1)
                        module = importlib.import_module(
                            "infinigen.assets.objects." + module_name
                        )
                        class_obj = getattr(module, class_name)
                        gen_class = class_obj
                    search_rels = filter_domain.relations
                    # 筛选出有效的关系，只选择非否定关系
                    search_rels = [
                        rd
                        for rd in search_rels
                        if not isinstance(rd[0], cl.NegatedRelation)
                    ]
                    if parent_obj_name == "1607620_RackFactory":
                        a = 1
                    assign = propose_relations.find_given_assignments(
                        self.state, search_rels, parent_obj_name=parent_obj_name
                    )
                    assignments = list(assign)
                    if len(assignments) == 0:
                        assignments = assignments
                    else:
                        assignments = assignments[0]
                    for i in range(2):
                        # for i, assignments in enumerate(assign):
                        found_tags = usage_lookup.usages_of_factory(gen_class)
                        move = moves.Addition(
                            names=[
                                f"{np.random.randint(1e6):04d}_{gen_class.__name__}"
                            ],  # decided later # 随机生成一个名称，基于生成器类的名称
                            gen_class=gen_class,  # 使用传入的生成器类
                            relation_assignments=assignments,  # 传入分配的关系
                            temp_force_tags=found_tags,  # 临时强制标签
                        )
                        target_name = class_name.lower()
                        if target_name.endswith("factory") or target_name.endswith(
                            "Factory"
                        ):
                            target_name = target_name[:-7]

                        target_name = f"{np.random.randint(1e7)}_{class_name}"
                        meshpath = None
                        success = move.apply_init(
                            self.state,
                            target_name,
                            size,
                            position,
                            rotation,
                            gen_class,
                            meshpath,
                        )
                        if not success:
                            self.delete_object(target_name)
                            if i == 1:
                                break
                            else:
                                assignments = [
                                    rel
                                    for rel in assignments
                                    if rel.target_name == "newroom_0-0"
                                ]
                                continue
                        else:
                            Placement[key][num]["name"] = target_name
                            break
                    # invisible_others()
                    # bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
                    # visible_others()

        return self.state

    @gin.configurable
    def init_graph_metascene(
        self,
        # filter_domain: r.Domain,
        var_assignments: dict[str, str],
        stage="large",  # large, medium, small
    ):
        relation_file = (
            "/mnt/fillipo/huangyue/recon_sim/7_anno_v3/metadata_support_yandan_v4.json"
        )
        with open(relation_file, "r") as f:
            scene_relations = json.load(f)

        def load_scene_relations(scene_id):
            # v4
            relations = scene_relations[scene_id]
            rel_small2big = dict()
            if relations == {}:
                return {}
            # for key in ["support","embed"]:
            for key in ["support"]:
                if key not in relations or relations[key] == {}:
                    continue
                for big_obj_id in relations[key]:
                    small_obj_ids = relations[key][big_obj_id]
                    big_id = big_obj_id.split("_")[0]
                    for small_id in small_obj_ids:
                        if small_id not in rel_small2big:
                            rel_small2big[small_id] = big_id
                        else:
                            AssertionError
            return rel_small2big

        scene_id = os.getenv("JSON_RESULTS")

        basedir = (
            f"/mnt/fillipo/huangyue/recon_sim/7_anno_v4/export_stage2_sm/{scene_id}/"
        )
        metadata = (
            f"/mnt/fillipo/yandan/metascene/export_stage2_sm/{scene_id}/metadata.json"
        )

        rel_small2big = load_scene_relations(scene_id)
        # PATH_TO_SCENES = os.getenv("JSON_RESULTS")
        # with open(PATH_TO_SCENES,"r") as f:
        #     Placement = json.load(f)
        with open(metadata, "r") as f:
            Placement = json.load(f)
        for step in ["large", "small"]:
            for key, value in Placement.items():
                if key == "5":
                    a = 1
                position = [0, 0, 0]
                rotation = 0
                size = None
                name = key
                if step == "large":
                    if name not in rel_small2big:  # deal with large object first
                        parent_obj_name = None
                        against_wall = False
                        on_floor = True
                        relation = None
                    else:
                        continue
                elif step == "small":
                    if name in rel_small2big:
                        on_floor = False
                        against_wall = False
                        parent_key = rel_small2big[name]
                        try:
                            parent_obj_name = Placement[parent_key]["target_name"]
                        except:
                            # failed in loading parent
                            continue
                        relation = "ontop"
                    else:
                        continue
                else:
                    AssertionError

                category = value["category"]
                if category in ["wall", "ceiling", "floor", "window"]:
                    continue

                # if stage == "small":
                #     this_stage = "small"
                #     parent_key,parent_num, relation = value[num]["parent"]
                #     parent_obj_name = self.Placement_big[parent_key][parent_num]["name"]
                #     against_wall = False
                #     on_floor = False
                #     size = [-1,-1,-1]
                # else:
                #     if "parent" in value[num]:
                #         this_stage = "medium"
                #         if this_stage!=stage:
                #             continue
                #         parent_key,parent_num, relation = value[num]["parent"]
                #         parent_obj_name = self.Placement_big[parent_key][parent_num]["name"]
                #     else:
                #         this_stage = "large"
                #         if this_stage!=stage:
                #             continue
                #         parent_obj_name = None

                #     against_wall = True if key in self.category_against_wall else False
                #     on_floor = True

                filter_domain = self.calc_filter_domain(
                    category,
                    num=None,
                    on_floor=on_floor,
                    against_wall=against_wall,
                    parent_obj_name=parent_obj_name,
                    relation=relation,
                )

                gen_class = copy.deepcopy(GeneralMetaFactory)
                size = None
                # x_dim, y_dim, z_dim = size

                # gen_class.x_dim = x_dim
                # gen_class.y_dim = y_dim
                # gen_class.z_dim = z_dim
                gen_class._category = category
                gen_class._asset_file = f"{basedir}/{key}.glb"
                front_view_angle = (
                    value["front_view"].split("/")[-1].split(".")[0].split("_")[-1]
                )
                angle_bias = (
                    value["front_view"].split("/")[-1].split(".")[0].split("_")[1]
                )
                gen_class._front_view_angle = int(front_view_angle) + int(angle_bias)
                class_name = category

                search_rels = filter_domain.relations
                # 筛选出有效的关系，只选择非否定关系
                search_rels = [
                    rd
                    for rd in search_rels
                    if not isinstance(rd[0], cl.NegatedRelation)
                ]

                found_tags = usage_lookup.usages_of_factory(gen_class)
                if search_rels[0][0].__class__.__name__ == "AnyRelation":
                    assignments = None
                else:
                    assign = propose_relations.find_given_assignments(
                        self.state, search_rels, parent_obj_name=parent_obj_name
                    )
                    for i, assignments in enumerate(assign):
                        break

                move = moves.Addition(
                    names=[
                        f"{np.random.randint(1e6):04d}_{gen_class.__name__}"
                    ],  # decided later # 随机生成一个名称，基于生成器类的名称
                    gen_class=gen_class,  # 使用传入的生成器类
                    relation_assignments=assignments,  # 传入分配的关系
                    temp_force_tags=found_tags,  # 临时强制标签
                )

                target_name = f"{np.random.randint(1e7)}_{class_name}"

                asset_file = f"{basedir}/{key}.glb"

                move.apply_init(
                    self.state,
                    target_name,
                    size,
                    position,
                    rotation,
                    gen_class,
                    asset_file,
                )
                if key == "5":
                    a = 1
                Placement[key]["target_name"] = target_name

                # invisible_others()
                # bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
                # visible_others()

        return self.state

    @gin.configurable
    def init_graph_physcene(
        self,
        # filter_domain: r.Domain,
        var_assignments: dict[str, str],
        stage="large",  # large, medium, small
    ):
        json_name = os.getenv("JSON_RESULTS")

        with open(json_name, "r") as f:
            Placement = json.load(f)
        for objname, obj_lst in Placement["ThreedFront"].items():
            for obj_info in obj_lst:
                category = obj_info["label"]
                if "lamp" in category:
                    continue

                position = obj_info["position"]
                position = [position[0], position[2], position[1]]
                radians = math.radians(90)
                rotation = radians - obj_info["theta"]
                scale = obj_info["scale"]

                name = category
                module_and_class = "ThreeDFuture"
                parent_obj_name = None
                against_wall = False
                on_floor = True

                filter_domain = self.calc_filter_domain(
                    category, num=None, on_floor=on_floor, against_wall=against_wall
                )

                gen_class = copy.deepcopy(GeneralThreedFrontFactory)
                gen_class._category = category
                gen_class._asset_file = obj_info["path"]
                gen_class._scale = scale
                gen_class._rotation = rotation
                gen_class._position = position
                class_name = category

                search_rels = filter_domain.relations
                # 筛选出有效的关系，只选择非否定关系
                search_rels = [
                    rd
                    for rd in search_rels
                    if not isinstance(rd[0], cl.NegatedRelation)
                ]

                assign = propose_relations.find_given_assignments(
                    self.state, search_rels, parent_obj_name=parent_obj_name
                )
                for i, assignments in enumerate(assign):
                    found_tags = usage_lookup.usages_of_factory(gen_class)
                    move = moves.Addition(
                        names=[
                            f"{np.random.randint(1e6):04d}_{gen_class.__name__}"
                        ],  # decided later # 随机生成一个名称，基于生成器类的名称
                        gen_class=gen_class,  # 使用传入的生成器类
                        relation_assignments=assignments,  # 传入分配的关系
                        temp_force_tags=found_tags,  # 临时强制标签
                    )

                    target_name = f"{np.random.randint(1e7)}_{class_name}"
                    while target_name in self.state.objs:
                        target_name = f"{np.random.randint(1e7)}_{class_name}"

                    if "lounge_chair" in target_name:
                        a = 1
                    # target_name = np.random.randint(1e7)+"_SofaFactory"

                    asset_file = obj_info["path"]

                    move.apply_init(
                        self.state,
                        target_name,
                        None,
                        position,
                        rotation,
                        gen_class,
                        asset_file,
                    )

                    # Placement[key][num]["name"] = target_name
                    break
                # invisible_others()
                # bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
                # visible_others()

        return self.state

    def transform_acdc(self, transform_info):
        # target
        # Create transformation matrices
        translation = Matrix.Translation(
            transform_info["target"]["location"]
        )  # Move 2 units in X, 3 in Y, 4 in Z
        rotation = Matrix.Rotation(
            transform_info["target"]["rotation"][-1], 4, "Z"
        )  # Rotate 90° around Z-axis (1.57 radians)
        scaling = Matrix.Diagonal(
            transform_info["target"]["scale"] + [1]
        )  # Scale by 1.5 in all axes
        # Combine transformations (Order: Scaling → Rotation → Translation)
        target_matrix = translation @ rotation @ scaling  # Matrix multiplication

        # source
        # Create transformation matrices
        translation = Matrix.Translation(
            transform_info["source"]["location"]
        )  # Move 2 units in X, 3 in Y, 4 in Z
        rot = transform_info["source"]["rotation"][-1] % (2 * math.pi)
        if rot < math.pi:
            rot = rot - math.pi
        rotation = Matrix.Rotation(
            rot, 4, "Z"
        )  # Rotate 90° around Z-axis (1.57 radians)
        scaling = Matrix.Diagonal(
            transform_info["source"]["scale"] + [1]
        )  # Scale by 1.5 in all axes
        # Combine transformations (Order: Scaling → Rotation → Translation)
        source_matrix = translation @ rotation @ scaling  # Matrix multiplication

        # obj
        # Create transformation matrices
        translation = Matrix.Translation(
            transform_info["obj"]["location"]
        )  # Move 2 units in X, 3 in Y, 4 in Z
        rotation = Matrix.Rotation(
            transform_info["obj"]["rotation"][-1], 4, "Z"
        )  # Rotate 90° around Z-axis (1.57 radians)
        scaling = Matrix.Diagonal(
            transform_info["obj"]["scale"] + [1]
        )  # Scale by 1.5 in all axes
        # Combine transformations (Order: Scaling → Rotation → Translation)
        obj_matrix = translation @ rotation @ scaling  # Matrix multiplication

        # merge
        obj_matrix_new = target_matrix @ source_matrix.inverted() @ obj_matrix
        # decompose
        location, rotation_quat, scale = obj_matrix_new.decompose()
        euler_rotation = rotation_quat.to_euler("XYZ")
        # print("Rotation (Euler):", euler_rotation)

        location = list(location)
        return location, euler_rotation, list(scale)

    def load_acdc(self, parent_obj_name="9577433_tv_stand"):
        transform_info = dict()
        # target_obj = self.state.objs[parent_obj_name].obj
        target_obj = bpy.data.objects.get(self.state.objs[parent_obj_name].populate_obj)

        transform_info["target"] = {
            "location": target_obj.location,
            "rotation": target_obj.rotation_euler,
            "scale": list(target_obj.scale),
            "size": target_obj.dimensions,
        }

        PATH_TO_SCENES = os.getenv("JSON_RESULTS")
        with open(PATH_TO_SCENES, "r") as f:
            scene_info = json.load(f)



        supporter = scene_info["supporter"]

        Placement = scene_info["objects"]

        for objname, obj_info in Placement.items():
            if objname == supporter:
                position_supporter = obj_info["location"]
                rotation_supporter = obj_info["rotation"][-1]
                scale_supporter = obj_info["scale"]
                size_supporter = obj_info["size"]
                transform_info["source"] = {
                    "location": position_supporter,
                    "rotation": obj_info["rotation"],
                    "scale": scale_supporter,
                    "size": size_supporter,
                }

        for objname, obj_info in Placement.items():
            if objname == supporter:
                continue
            category = "_".join(obj_info["category"].split("_")[:-1])
            position = obj_info["location"]
            rotation = obj_info["rotation"][-1]
            scale = obj_info["scale"]
            size = obj_info["size"]

            transform_info["obj"] = {
                "location": position,
                "rotation": obj_info["rotation"],
                "scale": scale,
                "size": size,
            }

            location_new, rotation_new, scale_new = self.transform_acdc(transform_info)

            gen_class = GeneralObjavFactory

            against_wall = False
            on_floor = False
            relation = "ontop"

            filter_domain = self.calc_filter_domain(
                category,
                num=None,
                on_floor=on_floor,
                against_wall=against_wall,
                parent_obj_name=parent_obj_name,
                relation=relation,
            )

            gen_class = GeneralObjavFactory
            x_dim, y_dim, z_dim = size
            gen_class._x_dim = x_dim * scale_new[0] / scale[0]
            gen_class._y_dim = y_dim * scale_new[1] / scale[1]
            gen_class._z_dim = z_dim * scale_new[2] / scale[2]
            gen_class._category = category
            gen_class._asset_file = obj_info["model"]
            gen_class._scale = scale
            gen_class._rotation = rotation_new[-1]
            gen_class._position = location_new
            class_name = category

            search_rels = filter_domain.relations
            # 筛选出有效的关系，只选择非否定关系
            search_rels = [
                rd for rd in search_rels if not isinstance(rd[0], cl.NegatedRelation)
            ]

            assign = propose_relations.find_given_assignments_fast(
                self.state, search_rels, parent_obj_name=parent_obj_name
            )
            for i, assignments in enumerate(assign):
                found_tags = usage_lookup.usages_of_factory(gen_class)
                move = moves.Addition(
                    names=[
                        f"{np.random.randint(1e6):04d}_{gen_class.__name__}"
                    ],  # decided later # 随机生成一个名称，基于生成器类的名称
                    gen_class=gen_class,  # 使用传入的生成器类
                    relation_assignments=assignments,  # 传入分配的关系
                    temp_force_tags=found_tags,  # 临时强制标签
                )

                target_name = f"{np.random.randint(1e7)}_{class_name}"
                while target_name in self.state.objs:
                    target_name = f"{np.random.randint(1e7)}_{class_name}"

                move.apply_init(
                    self.state,
                    target_name,
                    None,
                    location_new,
                    rotation_new[-1],
                    gen_class,
                    gen_class._asset_file,
                )

                break

        return self.state

    def get_bpy_objects(self, domain: r.Domain) -> list[bpy.types.Object]:
        objkeys = domain_contains.objkeys_in_dom(domain, self.state)
        return [self.state.objs[k].obj for k in objkeys]

    def calc_filter_domain(
        self,
        value,
        num=None,
        on_floor=True,
        against_wall=False,
        parent_obj_name=None,
        relation=None,
    ):
        if (
            num is not None
            and "parent" in value[num]
            and value[num]["parent"] != []
            and value[num]["parent"] is not None
        ):
            try:
                try:
                    parent_key, parent_num, relation = value[num]["parent"]
                    try:
                        parent_obj_name = self.Placement_big[parent_key][parent_num][
                            "name"
                        ]
                    except:
                        parent_obj_name = self.Placement[parent_key][parent_num]["name"]
                except:
                    parent_obj_name, relation = value[num]["parent"]

                var_assignments = {
                    cu.variable_room: "newroom_0-0",
                    cu.variable_obj: parent_obj_name,
                }
            except:
                parent_obj_name = None
                parent_key = None
                relation = None
                var_assignments = {cu.variable_room: "newroom_0-0"}

        elif parent_obj_name is not None:
            var_assignments = {
                cu.variable_room: "newroom_0-0",
                cu.variable_obj: parent_obj_name,
            }

        else:
            parent_obj_name = None
            parent_key = None
            relation = None
            var_assignments = {cu.variable_room: "newroom_0-0"}

        if (
            parent_obj_name is not None
            and parent_obj_name not in self.state.objs.keys()
        ):
            parent_obj_name = None
            parent_key = None
            relation = None
            var_assignments = {cu.variable_room: "newroom_0-0"}

        dom_assignments = {
            k: r.Domain(self.state.objs[objkey].tags)
            for k, objkey in var_assignments.items()
        }
        stage = self.get_stage(
            is_on_floor=on_floor,
            against_wall=against_wall,
            parent_obj_name=parent_obj_name,
            relation=relation,
        )

        filter_domain = r.substitute_all(stage, dom_assignments)

        return filter_domain

    def get_stage(self, is_on_floor, against_wall, parent_obj_name=None, relation=None):
        on_floor = cu.on_floor

        all_room = r.Domain({t.Semantics.Room, -t.Semantics.Object})
        all_obj = r.Domain({t.Semantics.Object, -t.Semantics.Room})
        all_obj_in_room = all_obj.with_relation(
            cl.AnyRelation(), all_room.with_tags(cu.variable_room)
        )
        primary = all_obj_in_room.with_relation(-cl.AnyRelation(), all_obj)
        secondary = all_obj.with_relation(
            cl.AnyRelation(), primary.with_tags(cu.variable_obj)
        )

        if parent_obj_name is not None and parent_obj_name != "newroom_0-0":
            module_name = (
                self.state.objs[parent_obj_name].generator.__module__
            )  #'infinigen.assets.threedfront_assets.threedfront_category'
            attribute_name = self.state.objs[
                parent_obj_name
            ].generator.__class__.__name__
            # Split into module name and attribute name
            # Dynamically import the module
            module = importlib.import_module(module_name)
            # Access the attribute (which could be a class, function, etc.)
            parent_Factory = getattr(module, attribute_name)

            # parent_Factory = self.state.objs[parent_obj_name].generator

            parent_domain = r.Domain(usage_lookup.usages_of_factory(parent_Factory))

            if (
                "sink" in parent_obj_name.lower()
                or "vanity" in parent_obj_name.lower()
                or "rack" in parent_obj_name.lower()
            ) and relation == "ontop":
                relation = "on"
            relation_module = getattr(cu, relation)
            stage = secondary.with_relation(relation_module, parent_domain)
        else:
            stage = primary

        if is_on_floor:
            stage = stage.with_relation(on_floor, all_room)
        if against_wall:
            stage = stage.with_relation(cu.against_wall, all_room)

        return stage

    @gin.configurable
    def init_graph_idesign(
        self,
    ):
        metadata = os.getenv("JSON_RESULTS")

        def get_obj_cnt(data):
            # Count types
            big_category_dict = {}

            for obj in data:
                if obj["new_object_id"] not in [
                    "south_wall",
                    "north_wall",
                    "east_wall",
                    "west_wall",
                    "middle of the room",
                    "ceiling",
                ]:
                    obj_type = "_".join(obj["new_object_id"].split("_")[:-1])
                    if obj_type not in big_category_dict:
                        big_category_dict[obj_type] = 0
                    big_category_dict[obj_type] += 1
            return big_category_dict

        with open(metadata, "r") as f:
            data = json.load(f)
            category_dict = get_obj_cnt(data)
            self.retrieve_objav_assets(category_dict)
            Placement = {}
            for item in data:
                if item["new_object_id"] not in [
                    "south_wall",
                    "north_wall",
                    "east_wall",
                    "west_wall",
                    "middle of the room",
                    "ceiling",
                ]:
                    Placement[item["new_object_id"]] = item

        # asset_dir = f"/mnt/fillipo/yandan/scenesage/idesign/scene_sage/assets_retrieve_by_IDesign/scene_{scene_idx}"
        for key, value in Placement.items():
            # name = key

            parent_obj_name = None
            against_wall = False
            on_floor = False
            relation = None

            category = "_".join(value["new_object_id"].split("_")[:-1])

            filter_domain = self.calc_filter_domain(
                category,
                num=None,
                on_floor=on_floor,
                against_wall=against_wall,
                parent_obj_name=parent_obj_name,
                relation=relation,
            )

            size = value["size_in_meters"]
            x_dim = size["width"]
            y_dim = size["length"]
            z_dim = size["height"]
            size = [x_dim, y_dim, z_dim]
            if "position" not in value:
                continue
            position = value["position"]
            position = [position["x"] + 0.14, position["y"] + 0.14, position["z"]]
            position[2] = position[2] - z_dim / 2 + 0.14
            rotation = (
                -(math.radians(value["rotation"]["z_angle"]) + math.pi) - math.pi / 2
            )
            # asset_file = f"{asset_dir}/{key}.glb"
            asset_file = self.LoadObjavFiles[category][0]

            gen_class = GeneralObjavFactory
            gen_class._x_dim = x_dim
            gen_class._y_dim = y_dim
            gen_class._z_dim = z_dim
            gen_class._category = category
            gen_class._asset_file = asset_file

            search_rels = filter_domain.relations
            # 筛选出有效的关系，只选择非否定关系
            search_rels = [
                rd for rd in search_rels if not isinstance(rd[0], cl.NegatedRelation)
            ]

            found_tags = usage_lookup.usages_of_factory(gen_class)
            if search_rels[0][0].__class__.__name__ == "AnyRelation":
                assignments = []  # TODO YYD None
            else:
                assign = propose_relations.find_given_assignments(
                    self.state, search_rels, parent_obj_name=parent_obj_name
                )
                for i, assignments in enumerate(assign):
                    break

            move = moves.Addition(
                names=[
                    f"{np.random.randint(1e6):04d}_{gen_class.__name__}"
                ],  # decided later # 随机生成一个名称，基于生成器类的名称
                gen_class=gen_class,  # 使用传入的生成器类
                relation_assignments=assignments,  # 传入分配的关系
                temp_force_tags=found_tags,  # 临时强制标签
            )

            target_name = f"{np.random.randint(1e7)}_{category}"
            try:
                move.apply_init(
                    self.state,
                    target_name,
                    size,
                    position,
                    rotation,
                    gen_class,
                    asset_file,
                )
            except:
                continue

            Placement[key]["target_name"] = target_name

            # invisible_others()
            # bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
            # visible_others()

        return self.state

    @gin.configurable
    def init_graph_layoutgpt(
        self,
    ):
        def get_obj_cnt(data):
            # Count types
            big_category_dict = {}

            for obj in data["objects"]:
                try:
                    obj_type = obj["type"]
                except:
                    obj_type = re.sub(r"[_]*\d+$", "", obj["new_object_id"])

                if obj_type not in big_category_dict:
                    big_category_dict[obj_type] = 0
                big_category_dict[obj_type] += 1
            return big_category_dict

        metadata = os.getenv("JSON_RESULTS")
        # scene_idx = metadata.split("_")[-1].split(".")[0]
        with open(metadata, "r") as f:
            data = json.load(f)
            category_dict = get_obj_cnt(data)
            self.retrieve_objav_assets(category_dict)

            Placement = {}
            for item in data["objects"]:
                Placement[item["new_object_id"]] = item

        for key, value in Placement.items():
            parent_obj_name = None
            against_wall = False
            on_floor = False
            relation = None

            category = re.sub(r"[_]*\d+$", "", key)

            filter_domain = self.calc_filter_domain(
                category,
                num=None,
                on_floor=on_floor,
                against_wall=against_wall,
                parent_obj_name=parent_obj_name,
                relation=relation,
            )

            size = value["size_in_meters"]
            x_dim = size["width"]
            y_dim = size["length"]
            z_dim = size["height"]
            size = [x_dim, y_dim, z_dim]
            # if "position" not in value:
            #     continue
            position = value["position"]
            position = [position["y"] + 0.14, position["x"] + 0.14, position["z"]]
            # position[2] = position[2] - z_dim / 2 + 0.14
            position[2] = position[2] + 0.14
            rotation = math.radians(value["rotation"]["z_angle"])  # + math.pi
            asset_file = self.LoadObjavFiles[category][0]

            gen_class = GeneralObjavFactory
            gen_class._x_dim = x_dim
            gen_class._y_dim = y_dim
            gen_class._z_dim = z_dim
            gen_class._category = category
            gen_class._asset_file = asset_file

            search_rels = filter_domain.relations
            # 筛选出有效的关系，只选择非否定关系
            search_rels = [
                rd for rd in search_rels if not isinstance(rd[0], cl.NegatedRelation)
            ]

            found_tags = usage_lookup.usages_of_factory(gen_class)
            if search_rels[0][0].__class__.__name__ == "AnyRelation":
                assignments = []  # TODO YYD None
            else:
                assign = propose_relations.find_given_assignments(
                    self.state, search_rels, parent_obj_name=parent_obj_name
                )
                for i, assignments in enumerate(assign):
                    break

            move = moves.Addition(
                names=[
                    f"{np.random.randint(1e6):04d}_{gen_class.__name__}"
                ],  # decided later # 随机生成一个名称，基于生成器类的名称
                gen_class=gen_class,  # 使用传入的生成器类
                relation_assignments=assignments,  # 传入分配的关系
                temp_force_tags=found_tags,  # 临时强制标签
            )

            target_name = f"{np.random.randint(1e7)}_{category}"
            try:
                move.apply_init(
                    self.state,
                    target_name,
                    size,
                    position,
                    rotation,
                    gen_class,
                    asset_file,
                )
            except:
                continue

            Placement[key]["target_name"] = target_name

            # invisible_others()
            # bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
            # visible_others()

        return self.state

    @gin.configurable
    def init_graph_atiss(
        self,
    ):
        json_name = os.getenv("JSON_RESULTS")

        with open(json_name, "r") as f:
            Placement = json.load(f)
        for objname, obj_lst in Placement["ThreedFront"].items():
            for obj_info in obj_lst:
                category = obj_info["label"]
                if "lamp" in category:
                    continue

                position = obj_info["position"]
                position = [position[0] + 0.14, position[2] + 0.14, position[1] + 0.14]
                radians = math.radians(90)
                rotation = radians - obj_info["theta"]
                scale = obj_info["scale"]

                name = category
                module_and_class = "ThreeDFuture"
                parent_obj_name = None
                against_wall = False
                on_floor = False

                filter_domain = self.calc_filter_domain(
                    category, num=None, on_floor=on_floor, against_wall=against_wall
                )

                gen_class = copy.deepcopy(GeneralThreedFrontFactory)
                gen_class._category = category
                gen_class._asset_file = obj_info["path"]
                gen_class._scale = scale
                gen_class._rotation = rotation
                gen_class._position = position
                class_name = category

                search_rels = filter_domain.relations
                # 筛选出有效的关系，只选择非否定关系
                search_rels = [
                    rd
                    for rd in search_rels
                    if not isinstance(rd[0], cl.NegatedRelation)
                ]
                found_tags = usage_lookup.usages_of_factory(gen_class)
                if search_rels[0][0].__class__.__name__ == "AnyRelation":
                    assignments = []  # TODO YYD None
                else:
                    assign = propose_relations.find_given_assignments(
                        self.state, search_rels, parent_obj_name=parent_obj_name
                    )
                    for i, assignments in enumerate(assign):
                        break

                move = moves.Addition(
                    names=[
                        f"{np.random.randint(1e6):04d}_{gen_class.__name__}"
                    ],  # decided later # 随机生成一个名称，基于生成器类的名称
                    gen_class=gen_class,  # 使用传入的生成器类
                    relation_assignments=assignments,  # 传入分配的关系
                    temp_force_tags=found_tags,  # 临时强制标签
                )

                target_name = f"{np.random.randint(1e7)}_{class_name}"
                while target_name in self.state.objs:
                    target_name = f"{np.random.randint(1e7)}_{class_name}"

                asset_file = obj_info["path"]

                move.apply_init(
                    self.state,
                    target_name,
                    None,
                    position,
                    rotation,
                    gen_class,
                    asset_file,
                )

                # Placement[key][num]["name"] = target_name
                break
                # invisible_others()
                # bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
                # visible_others()

        return self.state

    @gin.configurable
    def init_graph_anyhome(
        self,
    ):
        def get_obj_cnt(data):
            # Count types
            big_category_dict = {}

            for obj in data["objects"]:
                name = obj["new_object_id"]
                obj_type = re.sub(r"\d+$", "", name).lower()

                if obj_type not in big_category_dict:
                    big_category_dict[obj_type] = 0
                big_category_dict[obj_type] += 1
            return big_category_dict

        metadata = os.getenv("JSON_RESULTS")
        with open(metadata, "r") as f:
            data = json.load(f)
            category_dict = get_obj_cnt(data)
            self.retrieve_objav_assets(category_dict)

            Placement = {}
            for item in data["objects"]:
                Placement[item["new_object_id"]] = item

        for key, value in Placement.items():
            parent_obj_name = None
            against_wall = False
            on_floor = False
            relation = None

            category = re.sub(r"\d+$", "", value["new_object_id"]).lower()
            filter_domain = self.calc_filter_domain(
                category,
                num=None,
                on_floor=on_floor,
                against_wall=against_wall,
                parent_obj_name=parent_obj_name,
                relation=relation,
            )

            size = value["size_in_meters"]
            x_dim = size["width"]
            y_dim = size["length"]
            z_dim = size["height"]
            size = [x_dim, y_dim, z_dim]
            # if "position" not in value:
            #     continue
            position = value["position"]
            position = [position["x"] + 0.14, position["y"] + 0.14, position["z"]]
            position[2] = position[2] - z_dim / 2 + 0.14
            rotation = math.radians(value["rotation"]["z_angle"])  # + math.pi
            asset_file = self.LoadObjavFiles[category][0]

            gen_class = GeneralObjavFactory
            gen_class._x_dim = x_dim
            gen_class._y_dim = y_dim
            gen_class._z_dim = z_dim
            gen_class._category = category
            gen_class._asset_file = asset_file

            search_rels = filter_domain.relations
            # 筛选出有效的关系，只选择非否定关系
            search_rels = [
                rd for rd in search_rels if not isinstance(rd[0], cl.NegatedRelation)
            ]

            found_tags = usage_lookup.usages_of_factory(gen_class)
            if search_rels[0][0].__class__.__name__ == "AnyRelation":
                assignments = []  # TODO YYD None
            else:
                assign = propose_relations.find_given_assignments(
                    self.state, search_rels, parent_obj_name=parent_obj_name
                )
                for i, assignments in enumerate(assign):
                    break

            move = moves.Addition(
                names=[
                    f"{np.random.randint(1e6):04d}_{gen_class.__name__}"
                ],  # decided later # 随机生成一个名称，基于生成器类的名称
                gen_class=gen_class,  # 使用传入的生成器类
                relation_assignments=assignments,  # 传入分配的关系
                temp_force_tags=found_tags,  # 临时强制标签
            )

            target_name = f"{np.random.randint(1e7)}_{category}"
            try:
                move.apply_init(
                    self.state,
                    target_name,
                    size,
                    position,
                    rotation,
                    gen_class,
                    asset_file,
                )
            except:
                continue

            Placement[key]["target_name"] = target_name

            # invisible_others()
            # bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
            # visible_others()

        return self.state

    @gin.configurable
    def init_graph_holodeck(
        self,
    ):
        def get_obj_cnt(data):
            # Count types
            big_category_dict = {}

            for obj in data["objects"]:
                obj_type = (
                    obj["new_object_id"].split("-")[0].split("|")[0].replace(" ", "_")
                )
                if obj_type in ["window", "door"]:
                    continue
                if obj_type not in big_category_dict:
                    big_category_dict[obj_type] = 0
                big_category_dict[obj_type] += 1
            return big_category_dict

        metadata = os.getenv("JSON_RESULTS")

        with open(metadata, "r") as f:
            data = json.load(f)
            category_dict = get_obj_cnt(data)
            self.retrieve_objav_assets(category_dict)

            Placement = {}
            for item in data["objects"]:
                Placement[item["new_object_id"]] = item

        for key, value in Placement.items():
            parent_obj_name = None
            against_wall = False
            on_floor = False
            relation = None

            category = (
                value["new_object_id"].split("-")[0].split("|")[0].replace(" ", "_")
            )
            if category in ["window", "door"]:
                continue
            filter_domain = self.calc_filter_domain(
                category,
                num=None,
                on_floor=on_floor,
                against_wall=against_wall,
                parent_obj_name=parent_obj_name,
                relation=relation,
            )

            size = value["size_in_meters"]
            x_dim = size["width"]
            y_dim = size["length"]
            z_dim = size["height"]
            size = [x_dim, y_dim, z_dim]
            # if "position" not in value:
            #     continue
            position = value["position"]
            position = [position["x"] + 0.14, position["z"] + 0.14, position["y"]]
            position[2] = position[2] - z_dim / 2 + 0.14
            rotation = math.radians(90 - value["rotation"]["z_angle"])  # + math.pi
            asset_file = self.LoadObjavFiles[category][0]

            gen_class = GeneralObjavFactory
            gen_class._x_dim = x_dim
            gen_class._y_dim = y_dim
            gen_class._z_dim = z_dim
            gen_class._category = category
            gen_class._asset_file = asset_file

            search_rels = filter_domain.relations
            # 筛选出有效的关系，只选择非否定关系
            search_rels = [
                rd for rd in search_rels if not isinstance(rd[0], cl.NegatedRelation)
            ]

            found_tags = usage_lookup.usages_of_factory(gen_class)
            if search_rels[0][0].__class__.__name__ == "AnyRelation":
                assignments = []  # TODO YYD None
            else:
                assign = propose_relations.find_given_assignments(
                    self.state, search_rels, parent_obj_name=parent_obj_name
                )
                for i, assignments in enumerate(assign):
                    break

            move = moves.Addition(
                names=[
                    f"{np.random.randint(1e6):04d}_{gen_class.__name__}"
                ],  # decided later # 随机生成一个名称，基于生成器类的名称
                gen_class=gen_class,  # 使用传入的生成器类
                relation_assignments=assignments,  # 传入分配的关系
                temp_force_tags=found_tags,  # 临时强制标签
            )

            target_name = f"{np.random.randint(1e7)}_{category}"
            try:
                move.apply_init(
                    self.state,
                    target_name,
                    size,
                    position,
                    rotation,
                    gen_class,
                    asset_file,
                )
            except:
                continue

            Placement[key]["target_name"] = target_name

            # invisible_others()
            # bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
            # visible_others()

        return self.state
