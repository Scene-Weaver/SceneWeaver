import subprocess


def open_blender_for_render(blender_path, render_script, start_angle, end_angle):
    process = subprocess.Popen(
        [
            str(blender_path),
            "--background",
            "--python",
            str(render_script),
            str(start_angle),
            str(end_angle),
        ]
    )
    # subprocess.Popen([str(blender_path), '--background', '--python', str(render_script)])
    process.wait()
