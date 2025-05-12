system_prompt = """
You are given a list of objects in a 3D room scene, including their positions, rotations, sizes, and types (e.g., bed, desk, chair).
The unit is meters. The scene is also rendered from a top-down view.
Your task is to identify objects with incorrect, abnormal, or suboptimal sizes, and modify their dimensions to improve usability, proportion, and visual balance in the room.

For each object, check for size issues such as:
- Unrealistic or abnormal dimensions, e.g., a nightstand that is 2 meters tall, or a cup that is larger than a chair.
- Size incompatible with its supporting surface, e.g., an object on a table that is too large or small for the surface.
- Under- or over-scaled relative to the room or nearby furniture, e.g., a bed too small for the room, or a cabinet that blocks a window.
- Functional mismatch, e.g., a desk that is too small to use, or a TV too small to view from the bed.
- Lack of visual balance, e.g., shelves that are disproportionate to the wall or grouped items with inconsistent scaling.

For each object with a size issue:
- Briefly describe the problem.
- Suggest corrected dimensions (length, width, height in meters).
- Justify the correction based on real-world scale, spatial fit, or design logic.

You MUST modify object sizes when needed to improve realism and function. 
You should consider objects in different pose. For example, a laptop's size is different when closed and open. If the size match one of the pose, do not change the size.
You may also adjust the object’s position slightly to maintain spatial consistency after resizing.
You must not add or remove any object.
For objects that remain unchanged, include their original dimensions and layout in your response.

You are working in a 3D scene environment with the following conventions:

- Right-handed coordinate system.
- The X-Y plane is the floor.
- X axis (red) points right, Y axis (green) points top, Z axis (blue) points up.
- For the location [x,y,z], x,y means the location of object's center in x- and y-axis, z means the location of the object's bottom in z-axis.
- The size [sx,sy,sz] means [length, width, height] or [width, length, height], which depends on assets itself. 
- By default, assets face the +X direction.
- A rotation of [0, 0, 0.0] in Euler angles will turn the object to face +X.
- A rotation of [0, 0, 1.57] in Euler angles will turn the object to face +Y.
- A rotation of [0, 0, -1.57] in Euler angles will turn the object to face -Y.
- A rotation of [0, 0, 3.14] in Euler angles will turn the object to face -X.
- All bounding boxes are aligned with the local frame and marked in blue with category labels.
- The front direction of objects are marked with yellow arrow.
- Coordinates in the image are marked from [0, 0] at bottom-left of the room.

"""

user_prompt = """
Here is the information you receive:
1.This is a {roomtype}. 
2.The room size is [{roomsize}] in length and width.
3.User demand for the entire scene: {user_demand}
4.Ideas for this step (only for reference): {ideas}. You can refer to but not limited to the ideas.
5.This is the scene layout: {layout}
6.This is the layout of door and windows: {structure}
7.This is the image render from the top view: SCENE_IMAGE 

Please take a moment to relax and carefully look through each object's size. The unit is meters.
What size problem do you think it has? 
Then tell me how to solve these problems.

Before returning the final results, you need to carefully confirm that each issue has been resolved. 
If not, update the size until each problem is resolved.
For objects that remain unchanged, include their original dimensions and layout in your response.

Provide me with some explaination and the new layout of each object in json format.
Do not add any comment in the json. For example:
False:
"size": [0.3, 0.4,0.7],  // Adjusted to avoid overlap
True:
"size": [0.3, 0.4,0.7],
"""
