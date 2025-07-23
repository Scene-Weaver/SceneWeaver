import os

from app.agent.scenedesigner_noadd import SceneDesigner
from app.logger import logger


def main(prompt, i):
    agent = SceneDesigner()
    try:
        # prompt = "Design me a bedroom."
        save_dir = (
            "/mnt/fillipo/yandan/scenesage/record_scene/manus/0_kitchen_noadd/"
            + prompt[:30]
            .replace(" ", "_")
            .replace(".", "")
            .replace(",", "_")
            .replace("[", "")
            .replace("]", "")
        )

        save_dir = save_dir + "_" + str(i)
        os.system(
            f"cp {save_dir}/roominfo.json ../run/roominfo.json"
        )

        if not os.path.exists(save_dir):
            os.system(f"mkdir {save_dir}")
            os.system(f"mkdir {save_dir}/pipeline")
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
    # prompts = ["Design me an office.","Design me a classroom.","Design me a gym.","Design me a game room.","Design me a children room.",
    #    "Design me a classroom."]
    # prompts = ["Design me a small shop."] #
    # prompts = ["Design me a laundromat."]
    # prompts = ["Design me a single room of restaurant."]
    # prompts = ["Design me a bookstore."]
    # "Design me a meeting room."
    # "Design me a classroom."
    # "Design me a waiting room."
    # "Design me a clinic room."
    # "Design me an art studio.""
    # "Design me a kitchen."
    # "Design me a children room.", "Design me a game room.", "Design me a laboratory.","Design me a small bookstore.",
    # "Design me a waiting room.","Design me a laundry room."
    # "Design me a restaurant room.",
    # "Design me an office."
    prompts = ["Design me a kitchen."]  # computer room
    for p in prompts:
        for i in range(3):
            prompt = p
            main(prompt, i)
            # try:
            #     prompt = p
            #     main(prompt, i+2)
            # except:
            #     continue
    # import sys
    # prompt = sys.argv[-2]
    # i = sys.argv[-1]
    # main(prompt, i)
