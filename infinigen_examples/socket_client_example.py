#!/usr/bin/env python3
"""
Example client script for sending commands to the Blender socket server.
This script demonstrates how to communicate with the modified generate_indoors_mcp.py
"""

import argparse
import json
import socket


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


def main():
    parser = argparse.ArgumentParser(
        description="Send commands to Blender socket server"
    )
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=12345, help="Server port")
    parser.add_argument("--action", required=True, help="Action to perform")
    parser.add_argument("--iter", type=int, default=0, help="Iteration number")
    parser.add_argument("--description", default="", help="Description for the action")
    parser.add_argument("--save_dir", default="debug/", help="Save directory")
    parser.add_argument("--json_name", default="", help="JSON name")
    parser.add_argument("--inplace", default=False, help="Inplace flag")

    args = parser.parse_args()

    # Build command dictionary
    command = {
        "action": args.action,
        "iter": args.iter,
        "description": args.description,
        "save_dir": args.save_dir,
        "json_name": args.json_name,
        "inplace": args.inplace,
    }

    # Send command
    response = send_command(args.host, args.port, command)

    if response and response.get("status") == "queued":
        print(f"Command queued with ID: {response.get('command_id')}")
        print("The command will be processed by Blender...")


if __name__ == "__main__":
    # Example usage scenarios
    print("Example usage:")
    print("1. Check server status:")
    print("   python socket_client_example.py --action ping")
    print()
    print("2. Initialize physics scene:")
    print(
        "   python socket_client_example.py --action init_physcene --iter 0 --save_dir debug/"
    )
    print()
    print("3. Initialize metascene:")
    print(
        "   python socket_client_example.py --action init_metascene --iter 0 --save_dir debug/"
    )
    print()
    print("4. Stop server:")
    print("   python socket_client_example.py --action stop_server")
    print()

    # If no args provided, show examples
    import sys

    if len(sys.argv) == 1:
        print("Available actions:")
        print("- ping: Check if server is running")
        print("- status: Get server status and queue size")
        print("- init_physcene: Initialize physics scene")
        print("- init_metascene: Initialize metascene")
        print("- init_gpt: Initialize GPT")
        print("- stop_server: Stop the socket server")
        print()
        print("Use --help for full argument list")
    else:
        main()

# python infinigen_examples/socket_client_example.py --action add_gpt --iter 3 --save_dir /mnt/fillipo/yandan/scenesage/record_scene/manus/Design_me_a_printer_room_0 --json_name /mnt/fillipo/yandan/scenesage/record_scene/manus/Design_me_a_printer_room_0/pipeline/add_gpt_results_3.json
