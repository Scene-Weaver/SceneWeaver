# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

import copy
import logging
from pathlib import Path

import bpy
import gin
import numpy as np
from tqdm import trange

# from debug import invisible_others, visible_others
from infinigen.core.constraints import constraint_language as cl
from infinigen.core.constraints import reasoning as r
from infinigen.core.constraints.evaluator import domain_contains
from infinigen.core.constraints.example_solver import (
    greedy,
    propose_continous,
    propose_discrete,
)
from infinigen.core.constraints.example_solver.propose_discrete import moves
from infinigen.core.constraints.example_solver.state_def import State
from infinigen.core.util import blender as butil

from .annealing import SimulatedAnnealingSolver
from .room import MultistoryRoomSolver, RoomSolver
from infinigen.core.tags import Semantics, Subpart

from infinigen.assets.objaverse_assets import GeneralObjavFactory

from infinigen.assets.metascene_assets import GeneralMetaFactory
from infinigen.assets.threedfront_assets import GeneralThreedFrontFactory
from infinigen.assets.objects import (
            appliances,
            bathroom,
            decor,
            elements,
            lamp,
            seating,
            shelves,
            table_decorations,
            tables,
            tableware,
            wall_decorations,
        )
from infinigen.core.constraints import usage_lookup
import math
from . import moves, propose_relations, state_def

import importlib
from infinigen.core import tags as t
from infinigen_examples.util import constraint_util as cu
from infinigen_examples.util.visible import invisible_others, visible_others

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
        self.load_gpt_results()

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
        use_initial=False
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
            dom_assignments[cu.variable_obj] = r.Domain({Semantics.Object, -Semantics.Room})

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
        # #len(bounds) = 5

        if len(bounds) == 0:
            bounds = propose_discrete.preproc_bounds(
                orig_bounds, self.state, filter_domain, print_bounds=print_bounds
            )

            logger.info(f"No objects to be added for {desc_full=}, skipping")
            return self.state

        active_count = greedy.update_active_flags(self.state, var_assignments)  # 5,17

        n_start = len(self.state.objs)  # 37
        logger.info(
            f"Greedily solve {desc_full} - stage has {len(bounds)}/{len(orig_bounds)} bounds, "
            f"{active_count=}/{len(self.state.objs)} objs"
        )
        # [15:25:10.504] [solve] [INFO] | Greedily solve ('on_floor_0', 'bedroom_0-0') - stage has 7/63 bounds, active_count=5/25 objs
        # [10:56:47.439] [solve] [INFO] | Greedily solve ('obj_ontop_obj_0', 'bedroom_0-0', '195935_BedFactory') - stage has 2/63 bounds, active_count=17/37 objs

        self.optim.reset(max_iters=n_steps)
        # ra = (
        #     trange(n_steps) if self.optim.print_report_freq == 0 else range(n_steps)
        # )  # range(0, 150)
        ra = trange(n_steps) if self.optim.print_report_freq == 0 else range(n_steps)

        # 进行迭代
        for j in ra:
            # print(j)
            # if j==6:

            move_gen = self.choose_move_type(j, n_steps)  # 选择移动类型
            while move_gen.__name__ != "propose_translate" :#move_gen.__name__ != "propose_translate" and :
                move_gen = self.choose_move_type(j, n_steps) 
            
            # if desc == "on_floor_0":
            #     while move_gen.__name__ != "propose_translate" :#move_gen.__name__ != "propose_translate" and :
            #         move_gen = self.choose_move_type(j, n_steps)  # 选择移动类型
            # else:
            #     while move_gen.__name__ != "propose_relation_plane_change" :#move_gen.__name__ != "propose_translate" and :
            #         move_gen = self.choose_move_type(j, n_steps)  # 选择移动类型
            # print(move_gen , "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
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
        import json
        import os

        PATH_TO_SCENES = os.getenv("GPT_RESULTS")
        with open(PATH_TO_SCENES,"r") as f:
            info = json.load(f)
        self.name_mapping = info["name_mapping"]
        self.Placement_big = info["Placement_big"]
        self.Placement_small = info["Placement_small"]
        self.category_against_wall = info["category_against_wall"]

        return
    
    @gin.configurable
    def init_graph_gpt(
        self,
        # filter_domain: r.Domain,
        var_assignments: dict[str, str],
        stage = "large" #large, medium, small
    ):  
        if stage=="small":
            Placement = self.Placement_small
        else:
            Placement = self.Placement_big
        # dom_assignments = {
        #     k: r.Domain(self.state.objs[objkey].tags)
        #     for k, objkey in var_assignments.items()
        # }
        # filter_domains = dict()

        # stages_local = self.get_stages()
        # filter_domains["on_floor"] = r.substitute_all(stages_local["on_floor"], dom_assignments)
        # filter_domains["on_floor_against_wall"] = r.substitute_all(stages_local["on_floor_against_wall"], dom_assignments)
        
        for key, value in Placement.items():
            
            for num in value.keys():
                if num=='3':
                    a = 1
                
                position = value[num]["position"]
                if len(value[num]["position"])==2:
                    position += [0]
                rotation = value[num]["rotation"] * math.pi / 180
                size = value[num]["size"]
                name = key
                module_and_class = self.name_mapping[name]
                if stage == "small":
                    this_stage = "small"
                    parent_key,parent_num, relation = value[num]["parent"]
                    parent_obj_name = self.Placement_big[parent_key][parent_num]["name"]
                    against_wall = False
                    on_floor = False
                    size = [-1,-1,-1]
                else:
                    if "parent" in value[num]:
                        this_stage = "medium"
                        if this_stage!=stage:
                            continue
                        parent_key,parent_num, relation = value[num]["parent"]
                        parent_obj_name = self.Placement_big[parent_key][parent_num]["name"]
                    else:
                        this_stage = "large"
                        if this_stage!=stage:
                            continue
                        parent_obj_name = None

                    against_wall = True if key in self.category_against_wall else False
                    on_floor = True
                
                filter_domain = self.calc_filter_domain(value, num, on_floor=on_floor, against_wall=against_wall)

                if module_and_class is None:
                    gen_class = GeneralObjavFactory              
                    size = value[num]["size"]
                    x_dim, y_dim, z_dim = size
                    category = name
                    gen_class.x_dim = x_dim
                    gen_class.y_dim = y_dim
                    gen_class.z_dim = z_dim
                    gen_class.category = category

                    class_name = category
                else:
                    module_name, class_name = module_and_class.rsplit('.', 1)
                    module = importlib.import_module("infinigen.assets.objects."+module_name)
                    class_obj = getattr(module, class_name)
                    gen_class = class_obj
                search_rels = filter_domain.relations
                # 筛选出有效的关系，只选择非否定关系
                search_rels = [
                    rd for rd in search_rels if not isinstance(rd[0], cl.NegatedRelation)
                ]

                assign = propose_relations.find_given_assignments(self.state, search_rels, parent_obj_name=parent_obj_name)
                for i, assignments in enumerate(assign):
                    found_tags = usage_lookup.usages_of_factory( gen_class )  
                    move = moves.Addition(
                        names=[
                            f"{np.random.randint(1e6):04d}_{gen_class.__name__}"
                        ],  # decided later # 随机生成一个名称，基于生成器类的名称
                        gen_class=gen_class,  # 使用传入的生成器类
                        relation_assignments=assignments,  # 传入分配的关系
                        temp_force_tags=found_tags,  # 临时强制标签
                    )
                    
                    target_name = f"{np.random.randint(1e7)}_{class_name}"
                    # target_name = np.random.randint(1e7)+"_SofaFactory"
                    
                    meshpath = None

                    move.apply_init(
                        self.state, target_name, size, position, rotation, gen_class, meshpath
                    )

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
        stage = "large" #large, medium, small
    ):  
        basedir = "/mnt/fillipo/huangyue/recon_sim/7_anno_v4/export_stage2_sm/scene0001_00/"
        
        metadata = "/mnt/fillipo/yandan/metascene/export_stage2_sm/scene0001_00/metadata.json"
        import json
        with open(metadata,"r") as f:
            Placement = json.load(f)
        for key,value in Placement.items():
            category = value["category"]
            if category in ["wall","ceiling","floor"]:
                continue
            # if category == "floor":
            #     bpy.ops.import_scene.gltf(filepath=f"{basedir}/{key}.glb")
            #     imported_obj = bpy.context.selected_objects[0]
            #     bbox_local = imported_obj.bound_box
            #     import mathutils
            #     # Convert the bounding box corners to world space by applying the object's transformation
            #     bbox_world = [imported_obj.matrix_world @ mathutils.Vector(corner) for corner in bbox_local]

            
            position = [0,0,0]

            rotation = 0
            size = None
            name = key
            module_and_class = "MetaScene"
            parent_obj_name = None
            against_wall = False
            on_floor = True

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
            
            filter_domain = self.calc_filter_domain(category, num=None, on_floor=on_floor, against_wall=against_wall)

           
            gen_class = copy.deepcopy(GeneralMetaFactory)
            size = None
            # x_dim, y_dim, z_dim = size
            
            # gen_class.x_dim = x_dim
            # gen_class.y_dim = y_dim
            # gen_class.z_dim = z_dim
            gen_class._category = category
            gen_class._asset_file = f"{basedir}/{key}.glb"
            front_view_angle = value["front_view"].split("/")[-1].split(".")[0].split("_")[-1]
            gen_class._front_view_angle = int(front_view_angle)
            class_name = category
            
            search_rels = filter_domain.relations
            # 筛选出有效的关系，只选择非否定关系
            search_rels = [
                rd for rd in search_rels if not isinstance(rd[0], cl.NegatedRelation)
            ]

            assign = propose_relations.find_given_assignments(self.state, search_rels, parent_obj_name=parent_obj_name)
            for i, assignments in enumerate(assign):
                found_tags = usage_lookup.usages_of_factory( gen_class )  
                move = moves.Addition(
                    names=[
                        f"{np.random.randint(1e6):04d}_{gen_class.__name__}"
                    ],  # decided later # 随机生成一个名称，基于生成器类的名称
                    gen_class=gen_class,  # 使用传入的生成器类
                    relation_assignments=assignments,  # 传入分配的关系
                    temp_force_tags=found_tags,  # 临时强制标签
                )
                
                target_name = f"{np.random.randint(1e7)}_{class_name}"
                # target_name = np.random.randint(1e7)+"_SofaFactory"
                
                asset_file = f"{basedir}/{key}.glb"

                move.apply_init(
                    self.state, target_name, size, position, rotation, gen_class, asset_file
                )

                # Placement[key][num]["name"] = target_name
                break
            # invisible_others()
            # bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
            # visible_others()
                
        return self.state
    
    @gin.configurable
    def init_graph_physcene(
        self,
        # filter_domain: r.Domain,
        var_assignments: dict[str, str],
        stage = "large" #large, medium, small
    ):  
        basedir = "/home/yandan/workspace/PhyScene/3D_front/generate_filterGPN_clean"
        
        metadata = f"{basedir}/LivingDiningRoom-2954_livingroom.json"
        import json
        with open(metadata,"r") as f:
            Placement = json.load(f)
        for objname, obj_lst in Placement["ThreedFront"].items():
            for obj_info in obj_lst:
            
                category = obj_info["label"]
                if "lamp" in category:
                    continue

                position = obj_info["position"]
                position = [position[0],position[2],position[1]]
                radians = math.radians(90)
                rotation = radians-obj_info["theta"]
                scale = obj_info["scale"]

                name = category
                module_and_class = "ThreeDFuture"
                parent_obj_name = None
                against_wall = False
                on_floor = True

            
                filter_domain = self.calc_filter_domain(category, num=None, on_floor=on_floor, against_wall=against_wall)

            
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
                    rd for rd in search_rels if not isinstance(rd[0], cl.NegatedRelation)
                ]

                assign = propose_relations.find_given_assignments(self.state, search_rels, parent_obj_name=parent_obj_name)
                for i, assignments in enumerate(assign):
                    found_tags = usage_lookup.usages_of_factory( gen_class )  
                    move = moves.Addition(
                        names=[
                            f"{np.random.randint(1e6):04d}_{gen_class.__name__}"
                        ],  # decided later # 随机生成一个名称，基于生成器类的名称
                        gen_class=gen_class,  # 使用传入的生成器类
                        relation_assignments=assignments,  # 传入分配的关系
                        temp_force_tags=found_tags,  # 临时强制标签
                    )
                    
                    target_name = f"{np.random.randint(1e7)}_{class_name}"
                    # target_name = np.random.randint(1e7)+"_SofaFactory"
                    
                    asset_file = obj_info["path"]

                    move.apply_init(
                        self.state, target_name, None, position, rotation, gen_class, asset_file
                    )

                    # Placement[key][num]["name"] = target_name
                    break
                # invisible_others()
                # bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
                # visible_others()
                
        return self.state
    
    def get_bpy_objects(self, domain: r.Domain) -> list[bpy.types.Object]:
        objkeys = domain_contains.objkeys_in_dom(domain, self.state)
        return [self.state.objs[k].obj for k in objkeys]

    def calc_filter_domain(self, value, num=None, on_floor=True, against_wall=False):
        if num is not None and "parent" in value[num]:
            parent_key,parent_num, relation = value[num]["parent"]
            parent_obj_name = self.Placement_big[parent_key][parent_num]["name"]
            var_assignments = {cu.variable_room: 'newroom_0-0',
                                cu.variable_obj: parent_obj_name}            
        else:
            parent_obj_name = None
            parent_key = None
            relation = None
            var_assignments = {cu.variable_room: 'newroom_0-0'}    

        dom_assignments = {
            k: r.Domain(self.state.objs[objkey].tags)
            for k, objkey in var_assignments.items()
        }
        stage = self.get_stage(is_on_floor=on_floor, against_wall=against_wall, parent=parent_key, relation=relation)
        filter_domain = r.substitute_all(stage, dom_assignments)

        return filter_domain
         


    def get_stage(self, is_on_floor, against_wall, parent=None, relation=None):
    
        import importlib
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

        if parent is not None:
            parent_module = self.name_mapping[parent]
            # Split into module name and attribute name
            module_name, attribute_name = parent_module.rsplit('.', 1)
            # Dynamically import the module
            module = importlib.import_module("infinigen.assets.objects."+module_name)
            # Access the attribute (which could be a class, function, etc.)
            parent_Factory = getattr(module, attribute_name)
            parent_domain = r.Domain(usage_lookup.usages_of_factory(parent_Factory) )
            relation_module = getattr(cu, relation)
            stage = secondary.with_relation(relation_module, parent_domain)
        else:
            stage = primary

        if is_on_floor:
            stage = stage.with_relation(on_floor, all_room)
        if against_wall:
            stage = stage.with_relation(cu.against_wall, all_room)

        
        
        return stage
    


