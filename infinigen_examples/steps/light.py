import bpy

from infinigen.core.util import blender as butil


def turn_off(p):
    def turn_off_lights():
        for o in bpy.data.objects:
            if o.type == "LIGHT" and not o.data.cycles.is_portal:
                print(f"Deleting {o.name}")
                butil.delete(o)

    p.run_stage("lights_off", turn_off_lights)

    return
