import os

from app.agent.scenedesigner import SceneDesigner
from app.logger import logger


def main(prompt, i, basedir):
    agent = SceneDesigner()
    try:
        # prompt = "Design me a bedroom."
        save_dir = basedir + prompt[
            :30
        ].replace(" ", "_").replace(".", "").replace(",", "_").replace("[", "").replace(
            "]", ""
        )
        save_dir = save_dir + "_" + str(i)
        if not os.path.exists(save_dir):
            os.system(f"mkdir {save_dir}")
            

        os.makedirs(f"{save_dir}/pipeline", exist_ok=True)
        os.makedirs(f"{save_dir}/args", exist_ok=True)
        os.makedirs(f"{save_dir}/record_files", exist_ok=True)
        os.makedirs(f"{save_dir}/record_scene", exist_ok=True)
        os.environ["save_dir"] = save_dir
        os.environ["UserDemand"] = prompt
        if not prompt.strip():
            logger.warning("Empty prompt provided.")
            return

        logger.warning("Processing your request...")
        agent.run(prompt)
        logger.info("Request processing completed.")
    except KeyboardInterrupt:
        logger.warning("Operation interrupted.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--prompt', type=str,  default="Design me a bedroom.",
                       help='Your prompt to generate the scene. Default is "Design me a bedroom."')
    parser.add_argument('--cnt', type=int, default=1,
                   help='Number of scene to generate. Default is 1')
    parser.add_argument('--basedir', type=str, default="/mnt/fillipo/yandan/scenesage/record_scene/manus/",
                   help='The basic path to save all the generated scenes.')
    args = parser.parse_args()
    prompts = [args.prompt]
    cnt = args.cnt
    basedir = args.basedir
    
    # cnt = 3
    # prompts = ["Design me a baby room."]

    for p in prompts:
        for i in range(cnt):
            prompt = p
            main(prompt, i, basedir)
            