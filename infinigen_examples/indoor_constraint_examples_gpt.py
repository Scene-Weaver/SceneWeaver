# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

from collections import OrderedDict

from numpy.random import uniform

from infinigen.assets.objects import (
    lamp,
    seating,
    shelves,
    table_decorations,
    tables,
    tableware,
)
from infinigen.core.constraints import constraint_language as cl
from infinigen.core.constraints import usage_lookup
from infinigen.core.tags import Semantics

from .indoor_asset_semantics import home_asset_usage
from .util import constraint_util as cu


def sample_home_constraint_params():
    return dict(
        # what pct of the room floorplan should we try to fill with furniture?
        furniture_fullness_pct=uniform(0.6, 0.9),
        # how many objects in each shelving per unit of volume
        obj_interior_obj_pct=uniform(0.5, 1),  # uniform(0.6, 0.9),
        # what pct of top surface of storage furniture should be filled with objects? e.g pct of top surface of shelf
        obj_on_storage_pct=uniform(0.1, 0.9),
        # what pct of top surface of NON-STORAGE objects should be filled with objects? e.g pct of countertop/diningtable covered in stuff
        obj_on_nonstorage_pct=uniform(0.1, 0.6),
        # meters squared of wall art per approx meters squared of FLOOR area. TODO cant measure wall area currently.
        painting_area_per_room_area=uniform(20, 60) / 40,
        # rare objects wont even be added to the constraint graph in most homes
        has_tv=uniform() < 0.5,
        has_aquarium_tank=uniform() < 0.15,
        has_birthday_balloons=uniform() < 0.15,
        has_cocktail_tables=uniform() < 0.15,
        has_kitchen_barstools=uniform() < 0.15,
    )


def home_constraints():
    """Construct a constraint graph which incentivizes realistic home layouts.

    Result will contain both hard constraints (`constraints`) and soft constraints (`score_terms`).

    Notes for developers:
    - This function is typically evaluated ONCE. It is not called repeatedly during the optimization process.
        - To debug values you will need to inject print statements into impl_bindings.py or evaluate.py. Better debugging tools will come soon.
        - Similarly, most `lambda:` statements below will only be evaluated once to construct the graph - do not assume they will be re-evaluated during optimization.
    - Available constraint options are in `infinigen/core/constraints/constraint_language/__init__.py`.
        - You can easily add new constraint functions by adding them here, and defining evaluator functions for them in `impl_bindings.py`
        - Using newly added constraint types as hard constraints may be rejected by our hard constraint solver
    - It is quite easy to specify an impossible constraint program, or one that our solver cannot solve:
        - By default, failing to solve the program correctly is just printed as a warning, and we still return the scene.
        - You can cause failed optimization results to crash instead using `-p solve_objects.abort_unsatisfied=True` in the command line.
    - More documentation coming soon, and feel free to ask questions on Github Issues!

    """

    used_as = home_asset_usage()
    usage_lookup.initialize_from_dict(used_as)

    # rooms = cl.scene()[{Semantics.Room, -Semantics.Object}]
    # obj = cl.scene()[{Semantics.Object, -Semantics.Room}]

    # cutters = cl.scene()[Semantics.Cutter]
    # # window = cutters[Semantics.Window]
    # doors = cutters[Semantics.Door]

    # constraints = OrderedDict()
    # score_terms = OrderedDict()

    # furniture = obj[Semantics.Furniture].related_to(rooms, cu.on_floor)
    # wallfurn = furniture.related_to(rooms, cu.against_wall)
    # # storage = wallfurn[Semantics.Storage]

    # # Define specific categories relevant to a bookstore
    # bookshelves = wallfurn[shelves.LargeShelfFactory]
    # desks = furniture[shelves.SimpleDeskFactory]
    # reading_chairs = furniture[seating.OfficeChairFactory]
    # # decors = obj[decor.AquariumTankFactory]  # Decorative elements

    # # region Bookstore

    # # bookstores = rooms[Semantics.Utility].excludes(cu.room_types)

    # # Bookstore-specific rules
    # constraints["bookstore"] = rooms.all(
    #     lambda r: (
    #         bookshelves.related_to(r).count().in_range(3, 10)  # At least 3 shelves
    #         * reading_chairs.related_to(r).count().in_range(2, 6)  # Reading chairs available
    #         * desks.related_to(r).count().in_range(1, 2)  # One or two desks
    #         # * decors.related_to(r).count().in_range(0, 3)  # Optional decor
    #         # * bookshelves.all(
    #         #     lambda b: (
    #         #         cl.accessibility_cost(b, r, dist=1.5).minimize(weight=1)
    #         #         * cl.center_stable_surface_dist(b).minimize(weight=0.5)
    #         #     )
    #         # )
    #         # * reading_chairs.all(
    #         #     lambda c: (
    #         #         cl.accessibility_cost(c, bookshelves.related_to(r), dist=2).maximize(weight=1)
    #         #         * cl.focus_score(c, desks.related_to(r)).maximize(weight=1)
    #         #     )
    #         # )
    #     )
    # )

    # # Scoring for placement and aesthetic balance
    # score_terms["bookstore"] = rooms.mean(
    #     lambda r: (
    #         bookshelves.volume().maximize(weight=5)
    #         + bookshelves.related_to(r).mean(
    #             lambda b: (
    #                 b.distance(reading_chairs.related_to(r)).hinge(0.8, 1.2).minimize(weight=2)
    #                 + cl.angle_alignment_cost(b, r, cu.walltags).minimize(weight=1)
    #             )
    #         )
    #         + reading_chairs.related_to(r).mean(
    #             lambda c: (
    #                 c.distance(desks.related_to(r)).hinge(1.5, 2.0).minimize(weight=2)
    #                 + cl.focus_score(c, bookshelves.related_to(r)).maximize(weight=2)
    #             )
    #         )
    #         # + decors.volume().maximize(weight=1)
    #     )
    # )
    # # endregion

    # region base
    rooms = cl.scene()[{Semantics.Room, -Semantics.Object}]
    obj = cl.scene()[{Semantics.Object, -Semantics.Room}]

    constraints = OrderedDict()
    score_terms = OrderedDict()

    furniture = obj[Semantics.Furniture].related_to(rooms, cu.on_floor)
    wallfurn = furniture.related_to(rooms, cu.against_wall)
    storage = wallfurn[Semantics.Storage]

    params = sample_home_constraint_params()
    score_terms["furniture_fullness"] = rooms.mean(
        lambda r: (
            furniture.related_to(r)
            .volume(dims=(0, 1))
            .safediv(r.volume(dims=(0, 1)))
            .sub(params["furniture_fullness_pct"])
            .abs()
            .minimize(weight=15)
        )
    )

    # 通过计算房间内物品（如家具）与其他物品（如装饰物或容器）之间的体积比率来优化物品的排列，从而确保物品在物品中的填充度符合预期。
    score_terms["obj_in_obj_fullness"] = rooms.mean(
        lambda r: (
            furniture.related_to(r).mean(
                lambda f: (
                    obj.related_to(f, cu.on)
                    .volume()
                    .safediv(f.volume())
                    .sub(params["obj_interior_obj_pct"])
                    .abs()
                    .minimize(weight=10)  # 计算填充度误差并最小化
                )
            )
        )
    )

    def top_fullness_pct(f):
        return (
            obj.related_to(f, cu.ontop)
            .volume(dims=(0, 1))
            .safediv(f.volume(dims=(0, 1)))
        )

    score_terms["obj_ontop_storage_fullness"] = rooms.mean(
        lambda r: (
            storage.related_to(r).mean(
                lambda f: (
                    top_fullness_pct(f)
                    .sub(params["obj_on_storage_pct"])
                    .abs()
                    .minimize(weight=10)
                )
            )
        )
    )

    score_terms["obj_ontop_nonstorage_fullness"] = rooms.mean(
        lambda r: (
            furniture[-Semantics.Storage]
            .related_to(r)
            .mean(
                lambda f: (
                    top_fullness_pct(f)
                    .sub(params["obj_on_nonstorage_pct"])
                    .abs()
                    .minimize(weight=10)
                )
            )
        )
    )
    # endregion

    newroom = rooms[Semantics.NewRoom].excludes(cu.room_types)
    # region classroom

    # teacher_desk_obj = wallfurn[shelves.SimpleDeskFactory]
    # desk_obj = furniture[shelves.SimpleDeskFactory]#.related_to(teacher_desk_obj, cu.front_against)
    # bookshelf_obj = wallfurn[shelves.SimpleBookcaseFactory]
    # cabinet_obj = wallfurn[shelves.SingleCabinetFactory]
    # book_obj = obj[table_decorations.BookStackFactory]
    # laptop_obj = obj[appliances.MonitorFactory]

    # constraints["classroom"] = game_room.all(
    #     lambda r: (
    #         teacher_desk_obj.related_to(r).count().in_range(10, 10)
    #         * desk_obj.related_to(r).count().in_range(6, 6)
    #         * bookshelf_obj.related_to(r).count().in_range(2, 2)
    #         * cabinet_obj.related_to(r).count().in_range(1, 1)
    #         * bookshelf_obj.related_to(r).all(
    #             lambda s: (
    #                 book_obj.related_to(s, cu.on).count().in_range(5, 5)
    #                 * (book_obj.related_to(s, cu.on).count() >= 0)
    #             )
    #         )
    #         * desk_obj.related_to(r).all(
    #             lambda s: (
    #                 laptop_obj.related_to(s, cu.ontop).count().in_range(1, 1)
    #                 * (laptop_obj.related_to(s, cu.ontop).count() >= 0)
    #             )
    #         )
    #         * teacher_desk_obj.related_to(r).all(
    #             lambda s: (
    #                 laptop_obj.related_to(s, cu.ontop).count().in_range(1, 1)
    #                 * (laptop_obj.related_to(s, cu.ontop).count() >= 0)
    #             )
    #         )
    #     )
    # )

    # score_terms["kitchen_island_placement"] = game_room.mean(
    #     lambda r: (
    #         desk_obj.mean(
    #             lambda t: (
    #                 # cl.angle_alignment_cost(t, teacher_desk_obj)
    #                 # + cl.angle_alignment_cost(t, r, cu.walltags)
    #                 cl.angle_alignment_cost(
    #                     t, teacher_desk_obj.related_to(r), cu.front
    #                 ).minimize(weight=1)
    #             )
    #         ).minimize(weight=100)
    #     )
    # )
    # endregion

    # region diningroom
    # dining_table_obj = furniture[tables.TableDiningFactory]
    # dining_chairs_obj = furniture[seating.ChairFactory].related_to(dining_table_obj, cu.front_against)
    # cabinet_obj = wallfurn[shelves.SingleCabinetFactory]
    # plate_obj = obj[tableware.PlateFactory]
    # glass_obj = obj[tableware.CupFactory]
    # vase_obj = obj[table_decorations.VaseFactory]
    # bowl_obj = obj[tableware.BowlFactory]

    # constraints["dining_room"] = room.all(
    #     lambda r: (
    #         dining_table_obj.related_to(r).count().in_range(1, 1)
    #         * dining_chairs_obj.related_to(dining_table_obj.related_to(r)).count().in_range(6, 6)
    #         * cabinet_obj.related_to(r).count().in_range(1, 1)
    #         * dining_table_obj.related_to(r).all(
    #             lambda s: (
    #                 plate_obj.related_to(s, cu.ontop).count().in_range(6, 6)
    #                 * glass_obj.related_to(s, cu.ontop).count().in_range(6, 6)
    #                 * vase_obj.related_to(s, cu.ontop).count().in_range(1, 1)
    #                 * bowl_obj.related_to(s, cu.ontop).count().in_range(6, 6)
    #             )
    #         )
    #         * cabinet_obj.related_to(r).all(
    #             lambda s: (
    #                 plate_obj.related_to(s, cu.on).count().in_range(8, 8)
    #                 * glass_obj.related_to(s, cu.on).count().in_range(8, 8)
    #                 * bowl_obj.related_to(s, cu.on).count().in_range(8, 8)
    #             )
    #         )
    #     )
    # )
    # endregion

    # region living room
    Sofa_obj = furniture[seating.SofaFactory]
    CoffeeTable_obj = furniture[tables.CoffeeTableFactory]
    TVStand_obj = wallfurn[shelves.TVStandFactory]
    LargeShelf_obj = wallfurn[shelves.LargeShelfFactory]
    ArmChair_obj = furniture[seating.ArmChairFactory]
    SideTable_obj = furniture[tables.SideTableFactory]
    plant_obj = obj[tableware.PlantContainerFactory]
    vase_obj = obj[table_decorations.VaseFactory]
    books_obj = obj[table_decorations.BookStackFactory]
    # FloorLamp_obj = furn   iture[lamp.FloorLampFactory].related_to(ArmChair_obj,cu.side_by_side)
    FloorLamp_obj = (
        obj[lamp.FloorLampFactory].related_to(rooms, cu.on_floor)
    ).related_to(ArmChair_obj, cu.side_by_side)
    constraints["living_room"] = newroom.all(
        lambda r: (
            Sofa_obj.related_to(r).count().in_range(2, 2)
            * CoffeeTable_obj.related_to(r).count().in_range(1, 1)
            # * CoffeeTable_obj.related_to(Sofa_obj.related_to(r), cu.front_against).count().in_range(1, 1)
            # * TVStand_obj.related_to(r).count().in_range(1, 1)
            # * LargeShelf_obj.related_to(r).count().in_range(1, 1)
            # * LargeShelf_obj.related_to(r).all(
            #     lambda s: (
            #         plant_obj.related_to(s, cu.on).count().in_range(1, 1)
            #         * (plant_obj.related_to(s, cu.on).count() >= 0)
            #         * vase_obj.related_to(s, cu.on).count().in_range(1, 1)
            #         * (vase_obj.related_to(s, cu.on).count() >= 0)
            #         * books_obj.related_to(s, cu.on).count().in_range(3, 3)
            #         * (books_obj.related_to(s, cu.on).count() >= 0)
            #     )
            # )
            # * ArmChair_obj.related_to(r).count().in_range(2, 2)
            # * SideTable_obj.related_to(r).count().in_range(2, 2)
            # * SideTable_obj.related_to(Sofa_obj.related_to(r), cu.side_by_side).count().in_range(2, 2)
            # * FloorLamp_obj.related_to(r).count().in_range(2, 2)
            # * SideTable_obj.related_to(r).all(
            #     lambda s: (
            #         plant_obj.related_to(s, cu.ontop).count().in_range(1, 1)
            #         * (plant_obj.related_to(s, cu.ontop).count() >= 0)
            #     )
            # )
            # * CoffeeTable_obj.related_to(r).all(
            #     lambda s: (
            #         vase_obj.related_to(s, cu.ontop).count().in_range(1, 1)
            #         * (vase_obj.related_to(s, cu.ontop).count() >= 0)
            #     )
            # )
        )
    )
    # endregion
    # region bookstore
    # rooms = cl.scene()[{Semantics.Room, -Semantics.Object}]
    # obj = cl.scene()[{Semantics.Object, -Semantics.Room}]
    # bookstore = rooms[Semantics.Office].excludes(cu.room_types)

    # constraints = OrderedDict()
    # score_terms = OrderedDict()

    # furniture = obj[Semantics.Furniture].related_to(rooms, cu.on_floor)
    # wallfurn = furniture.related_to(rooms, cu.against_wall)

    # bookshelves_obj = wallfurn[shelves.LargeShelfFactory]
    # checkout_counter_obj = furniture[shelves.SidetableDeskFactory]
    # reading_tables_obj = furniture[tables.TableDiningFactory]
    # chairs_obj = furniture[seating.ChairFactory].related_to(
    #     reading_tables_obj, cu.front_against
    # )

    # books_obj = obj[table_decorations.BookStackFactory]
    # lamps_obj = obj[lamp.DeskLampFactory]

    # constraints["bookstore"] = bookstore.all(
    #     lambda r: (
    #         checkout_counter_obj.related_to(r).count().in_range(1, 1)
    #         * bookshelves_obj.related_to(r).count().in_range(5, 5)
    #         * reading_tables_obj.related_to(r).count().in_range(2, 2)
    #         * chairs_obj.related_to(r).count().in_range(8, 8)
    #         # * (chairs_obj.related_to(r).count()>=0)
    #         * bookshelves_obj.related_to(r).all(
    #             lambda s: (
    #                 books_obj.related_to(s, cu.on).count().in_range(10, 10)
    #                 # * (books_obj.related_to(s, cu.on).count()>=0)
    #             )
    #         )
    #         * reading_tables_obj.related_to(r).all(
    #             lambda t: (
    #                 books_obj.related_to(t, cu.ontop).count().in_range(5, 5)
    #                 * lamps_obj.related_to(t, cu.ontop).count().in_range(4, 4)
    #                 # * (lamps_obj.related_to(t, cu.ontop).count()>=0)
    #             )
    #         )
    #     )
    # )

    # score_terms["floor_covering"] = reading_tables_obj.mean(
    #     lambda r: (
    #         r.distance(rooms, cu.walltags).maximize(weight=3)
    #         * cl.angle_alignment_cost(r, rooms, cu.walltags).minimize(weight=3)
    #     )
    # )

    # # endregion

    # # region bedroom
    # rooms = cl.scene()[{Semantics.Room, -Semantics.Object}]
    # obj = cl.scene()[{Semantics.Object, -Semantics.Room}]

    # constraints = OrderedDict()
    # score_terms = OrderedDict()

    # furniture = obj[Semantics.Furniture].related_to(rooms, cu.on_floor)
    # wallfurn = furniture.related_to(rooms, cu.against_wall)

    # beds_obj = wallfurn[seating.BedFactory]
    # desks_obj = wallfurn[shelves.SimpleDeskFactory]
    # nightstand_obj = wallfurn[shelves.SingleCabinetFactory]

    # floor_lamps_obj = obj[lamp.FloorLampFactory].related_to(rooms, cu.on_floor).related_to(rooms, cu.against_wall)
    # books_obj = obj[table_decorations.BookStackFactory]

    # bedrooms = rooms[Semantics.Office].excludes(cu.room_types)
    # constraints["bedroom"] = bedrooms.all(
    #     lambda r: (
    #         beds_obj.related_to(r).count().in_range(1, 1)
    #         * (
    #             nightstand_obj.related_to(r)
    #             .related_to(beds_obj.related_to(r), cu.leftright_leftright)
    #             .count()
    #             .in_range(2, 2)
    #         )
    #         * desks_obj.related_to(r).count().in_range(1, 1)
    #         * floor_lamps_obj.related_to(r).count().in_range(1, 1)
    #         * nightstand_obj.related_to(r).all(
    #             lambda s: (
    #                 books_obj.related_to(s, cu.on).count() >= 0
    #             )
    #         )
    #     )
    # )
    # # endregion

    return cl.Problem(
        constraints=constraints,
        score_terms=score_terms,
    )


all_constraint_funcs = [home_constraints]
