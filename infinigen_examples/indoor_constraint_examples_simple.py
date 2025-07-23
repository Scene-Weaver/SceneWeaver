# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

from collections import OrderedDict

from numpy.random import uniform

from infinigen.assets.objects import (
    appliances,
    seating,
    shelves,
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

    rooms = cl.scene()[{Semantics.Room, -Semantics.Object}]
    obj = cl.scene()[{Semantics.Object, -Semantics.Room}]

    cutters = cl.scene()[Semantics.Cutter]
    window = cutters[Semantics.Window]
    doors = cutters[Semantics.Door]

    constraints = OrderedDict()
    score_terms = OrderedDict()

    # region overall fullness

    furniture = obj[Semantics.Furniture].related_to(rooms, cu.on_floor)
    wallfurn = furniture.related_to(rooms, cu.against_wall)
    storage = wallfurn[Semantics.Storage]

    # region OFFICES
    offices = rooms[Semantics.Office].excludes(cu.room_types)
    # desks_office = furniture[shelves.SimpleDeskFactory].related_to(rooms)
    desks_office = wallfurn[shelves.SimpleDeskFactory]
    deskchairs_office = furniture[seating.OfficeChairFactory].related_to(
        desks_office, cu.front_to_front
    )
    monitors_office = obj[appliances.MonitorFactory].related_to(desks_office, cu.ontop)
    # deskchairs_office = furniture[seating.OfficeChairFactory]

    constraints["office"] = offices.all(
        lambda r: (
            desks_office.related_to(r).count().in_range(6, 6)
            * desks_office.related_to(r).all(
                lambda t: (
                    (deskchairs_office.related_to(t).count().in_range(2, 2))
                    * (monitors_office.related_to(t).count().in_range(2, 2))
                    # * (obj[Semantics.OfficeShelfItem].related_to(t, cu.on).count() >= 3)
                    * (deskchairs_office.related_to(r).related_to(t).count() >= 0)
                    * (monitors_office.related_to(t).count() >= 0)
                )
            )
            # *(deskchairs_office.count()==2)
            # * (  # allow sidetables next to any sofa
            #     deskchairs_office.related_to(r)
            #     .related_to(sofas.related_to(r), cu.side_by_side)
            #     .count()
            #     .in_range(0, 2)
            # )
        )
    )
    # constraints["desk"] = desks_office.all(
    #     lambda t: (
    #         deskchairs_office.related_to(t).count()==1
    #     )
    # )

    # * (  # allow sidetables next to any sofa
    #             sidetable.related_to(r)
    #             .related_to(sofas.related_to(r), cu.side_by_side)
    #             .count()
    #             .in_range(0, 2)
    #         )
    #         * desks.related_to(r).count().in_range(0, 1)
    #         * coffeetables.related_to(r).count().in_range(0, 1)
    #         * coffeetables.related_to(r).all(
    #             lambda t: (
    #                 obj[Semantics.OfficeShelfItem]
    #                 .related_to(t, cu.on)
    #                 .count()
    #                 .in_range(0, 3)
    #             )
    #         )
    # constraints["chair"] = offices.all(
    #     lambda r: (
    #         # allow 0-2 lamps per room, placed on any sensible object
    #         (deskchairs_office.related_to(r).count()+1)==(desks_office.related_to(r).count()+1)
    #     )
    # )

    # # endregion
    # region DESKS
    # desks = wallfurn[shelves.SimpleDeskFactory]
    # deskchair = furniture[seating.OfficeChairFactory].related_to(
    #     desks, cu.front_against
    # )
    # monitors = obj[appliances.MonitorFactory]
    # constraints["desk"] = rooms.all(
    #     lambda r: (
    #         desks.related_to(r).all(
    #             lambda t: (
    #                 deskchair.related_to(r).related_to(t).count().in_range(1, 1)
    #                 # * monitors.related_to(t, cu.ontop).count().equals(1)
    #                 # * (obj[Semantics.OfficeShelfItem].related_to(t, cu.on).count() >= 0)
    #                 * (deskchair.related_to(r).related_to(t).count() == 1)
    #             )
    #         )
    #     )
    # )

    # score_terms["desk"] = rooms.mean(
    #     lambda r: desks.mean(
    #         lambda d: (
    #             obj.related_to(d).count().maximize(weight=3)
    #             + d.distance(doors.related_to(r)).maximize(weight=0.1)
    #             + cl.accessibility_cost(d, furniture.related_to(r)).minimize(weight=3)
    #             + cl.accessibility_cost(d, r).minimize(weight=3)
    #             + monitors.related_to(d).mean(
    #                 lambda m: (
    #                     cl.accessibility_cost(m, r, dist=2).minimize(weight=3)
    #                     + cl.accessibility_cost(
    #                         m, obj.related_to(r), dist=0.5
    #                     ).minimize(weight=3)
    #                     + m.distance(r, cu.walltags).hinge(0.1, 1e7).minimize(weight=1)
    #                 )
    #             )
    #             + deskchair.distance(rooms, cu.walltags).maximize(weight=1)
    #         )
    #     )
    # )

    # endregion
    # endregion

    # # region DESKS
    # desks = wallfurn[shelves.SimpleDeskFactory]
    # deskchair = furniture[seating.OfficeChairFactory].related_to(
    #     desks, cu.front_against
    # )
    # monitors = obj[appliances.MonitorFactory]
    # constraints["desk"] = rooms.all(
    #     lambda r: (
    #         (desks.related_to(r).count().in_range(4,4))
    #         *desks.related_to(r).all(
    #             lambda t: (
    #                 deskchair.related_to(r).related_to(t).count().in_range(2, 2)
    #                 * monitors.related_to(t, cu.ontop).count().in_range(2, 2)
    #                 # * (obj[Semantics.OfficeShelfItem].related_to(t, cu.on).count() >= 0)
    #                 * (monitors.related_to(t, cu.ontop).count() >= 2)
    #                 * (deskchair.related_to(r).related_to(t).count() >= 2)
    #             )
    #         )
    #     )
    # )

    # score_terms["desk"] = rooms.mean(
    #     lambda r: desks.mean(
    #         lambda d: (
    #             obj.related_to(d).count().maximize(weight=3)
    #             + d.distance(doors.related_to(r)).maximize(weight=0.1)
    #             + cl.accessibility_cost(d, furniture.related_to(r)).minimize(weight=3)
    #             + cl.accessibility_cost(d, r).minimize(weight=3)
    #             + monitors.related_to(d).mean(
    #                 lambda m: (
    #                     cl.accessibility_cost(m, r, dist=2).minimize(weight=3)
    #                     + cl.accessibility_cost(
    #                         m, obj.related_to(r), dist=0.5
    #                     ).minimize(weight=3)
    #                     + m.distance(r, cu.walltags).hinge(0.1, 1e7).minimize(weight=1)
    #                 )
    #             )
    #             + deskchair.distance(rooms, cu.walltags).maximize(weight=1)
    #         )
    #     )
    # )

    # endregion

    return cl.Problem(
        constraints=constraints,
        score_terms=score_terms,
    )


all_constraint_funcs = [home_constraints]
