from .blender_render import new_angle_range, render_rotation
from .feature_matcher import FeatureMatcher
from .open_blender import open_blender_for_render


def debug():
    fm = FeatureMatcher()
    fm.eval()

    blender_path = "~/workspace/SceneWeaver/blender/blender"
    render_script = "~/workspace/SceneWeaver/match/blender_render.py"
    start_angle = 0
    end_angle = 359

    # render_rotation("~/Desktop/dift/source.obj",0,360)

    input_rgb_path = "~/Desktop/dift/ref/standard.png"
    candidate_imgs_fdirs = "~/Desktop/dift/source"
    topk_model_candidates_dir = "~/Desktop/dift/top_k_model_candidates"
    name = "laptop"
    # i = 0

    for i in range(2):
        print(start_angle, end_angle)
        open_blender_for_render(blender_path, render_script, start_angle, end_angle)

        model_results = fm.find_nearest_neighbor_candidates(
            input_img_fpath=input_rgb_path,
            candidate_imgs_fdirs=candidate_imgs_fdirs,
            # candidate_imgs=candidate_imgs,
            candidate_filter=None,
            n_candidates=2,
            save_dir=topk_model_candidates_dir,
            visualize_resolution=(640, 480),
            save_prefix=f"{name}_iter{i}",
            remove_background=True,
        )

        current_candidates = model_results["candidates"]
        top_k_angles = [int(i.split("/")[-1].split(".")[0]) for i in current_candidates]
        start_angle, end_angle = new_angle_range(top_k_angles)
    return top_k_angles[0]
