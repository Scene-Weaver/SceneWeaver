import json
import os
import socket
import subprocess


def send_command(host="localhost", port=12345, command=None):
    """Send a single command to the Blender socket server"""
    try:
        # Create socket connection
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((host, port))

        # Send command
        command_json = json.dumps(command)
        client_socket.send(command_json.encode("utf-8"))

        # Receive response
        response = client_socket.recv(1024)
        response_data = json.loads(response.decode("utf-8"))

        print(f"Sent: {command}")
        print(f"Response: {response_data}")

        return response_data

    except Exception as e:
        print(f"Error: {e}")
        return None
    finally:
        if "client_socket" in locals():
            client_socket.close()


def update_infinigen(
    action,
    iter,
    json_name,
    ideas=None,
    description=None,
    inplace=False,
    invisible=False,
):
    j = {
        "iter": iter,
        "action": action,
        "json_name": json_name,
        #  "roomsize": roomsize,
        "description": description,
        "inplace": inplace,
        "success": False,
        "ideas": ideas,
    }
    save_dir = os.getenv("save_dir")
    argsfile = f"{save_dir}/args.json"
    with open(argsfile, "w") as f:
        json.dump(j, f, indent=4)
    os.system(
        f"cp {save_dir}/roominfo.json ../run/roominfo.json"
    )

    # # if invisible:
    if action == "export_supporter":
        # if True:
        cmd = f"""
        source ~/anaconda3/etc/profile.d/conda.sh
        conda activate infinigen_python
        cd ~/workspace/SceneWeaver
        python -m infinigen_examples.generate_indoors --seed 0 --save_dir {save_dir} --task coarse --output_folder outputs/indoors/coarse_expand_whole_nobedframe -g fast_solve.gin overhead.gin studio.gin -p compose_indoors.terrain_enabled=False compose_indoors.invisible_room_ceilings_enabled=True > ~/workspace/SceneWeaver/Pipeline/run.log 2>&1
        """
        subprocess.run(["bash", "-c", cmd])
        # else:
        #     os.system("bash -i ~/workspace/SceneWeaver/run.sh > run.log 2>&1")
    else:
        command = {
            "action": action,
            "iter": iter,
            "description": description,
            "save_dir": save_dir,
            "json_name": json_name,
            "inplace": inplace,
        }
        # Send command
        response = send_command("localhost", 12345, command)

    with open(argsfile, "r") as f:
        j = json.load(f)

    assert j["success"]
    print("infinigen success")
    return j["success"]
