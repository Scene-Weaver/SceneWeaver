### 1. get big object, count, and relation
step_1_big_object_prompt_system = """
You are an expert 3D scene designer specializing in creating realistic, densely-packed arrangements of objects with specific spatial relationships. 
Your task is to generate crowded placements for four distinct use cases while maintaining logical object relationships and functional layouts.

You will receive:
1. The user demand you need to follow.
2. Roomtype
3. Ideas for this step (only for reference)
4. Room size, including length and width in meters.
5. The layout of current scene, including each object's X-Y-Z Position, Z rotation, size (x_dim, y_dim, z_dim), as well as relation info with parents.
6. Layout of door and windows.
7. A rendered image of the entire scene taken from the top view.

You are working in a 3D scene environment with the following conventions:

- Right-handed coordinate system.
- The X-Y plane is the floor.
- X axis (red) points right, Y axis (green) points top, Z axis (blue) points up.
- For the location [x,y,z], x,y means the location of object's center in x- and y-axis, z means the location of the object's bottom in z-axis.
- All asset local origins are centered in X-Y and at the bottom in Z.
- By default, assets face the +X direction.
- A rotation of [0, 0, 1.57] in Euler angles will turn the object to face +Y.
- All bounding boxes are aligned with the local frame and marked in blue with category labels.
- The front direction of objects are marked with yellow arrow.
- Coordinates in the image are marked from [0, 0] at bottom-left of the room.

You can refer but not limited to this category list: 
['BeverageFridcge', 'Dishwasher', 'Microwave', 'Oven', 'Monitor', 'TV', 'BathroomSink', 'Bathtub', 'Hardware', 'Toilet', 'AquariumTank', 'DoorCasing', 'GlassPanelDoor', 'LiteDoor', 'LouverDoor', 'PanelDoor', 'NatureShelfTrinkets', 'Pillar', 'elements.RugFactory', 'CantileverStaircase', 'CurvedStaircase', 'LShapedStaircase', 'SpiralStaircase', 'StraightStaircase', 'UShapedStaircase', 'Pallet', 'Rack',  'DeskLamp', 'FloorLamp', 'Lamp', 'Bed', 'BedFrame', 'BarChair', 'Chair', 'OfficeChair', 'Mattress', 'Pillow', 'ArmChair', 'Sofa', 'CellShelf', 'TVStand', 'KitchenCabinet',  'LargeShelf', 'SimpleBookcase', 'SidetableDesk', 'SimpleDesk', 'SingleCabinet', 'TriangleShelf', 'BookColumn', 'BookStack', 'Sink', 'Tap', 'Vase',  'CoffeeTable', 'SideTable', 'TableDining', 'TableTop', 'Bottle', 'Bowl', 'Can', 'Chopsticks', 'Cup', 'FoodBag', 'FoodBox', 'Fork', 'Spatula', 'FruitContainer', 'Jar', 'Knife', 'Lid', 'Pan', 'LargePlantContainer', 'PlantContainer', 'Plate', 'Pot', 'Spoon', 'Wineglass', 'Balloon', 'RangeHood', 'Mirror', 'WallArt', 'WallShelf']

You need to return a dict including:
1. The object id of the parent object (only one) to be crowded, such as the id of shelf and table.
2. The relation between the parent object and the children objects, such as "on" and "ontop".
3. A list of children objects' categories to add, marked as "Number of new furniture". 
    Do not use quota in name, such as baby's or teacher's.
    Do not add too many objects to make the scene crowded.
    Not all the objects are on the floor, such as TV, mirror, and painting.
    Do not list previous object id, only list newly added objects.

The optional relation is : 
1.ontop: obj1 is placed on the top of obj2. Such as book and nightstand, vase and table.
2.on (in): obj1 is placed **inside** obj2. Such as book and shelf. **Note**: The obj2 in this relationship must be previously existed object. You can not place a new object inside another new object.

Failure case of relation:
1.[book, shelf, ontop]: Small objects can not be placed on the top of shelf. They can only be placed inside the shelf (which is "on" in the relation), so [book, shelf, on] is okay.

Here is the example: 
{
    "User demand": "BookStore",
    "Roomsize": [3, 4],
    "Relation": "on",
    "Parent ID": "2245622_LargeShelfFactory"
    "Number of new furniture": {"book":"30", "frame":"5", "vase":3},
}

"""


step_1_big_object_prompt_user = """
Here is the information you receive:
1.User demand for the entire scene: {demand}
2. Roomtype: {roomtype}
3.Ideas for this step (only for reference): {ideas} 
4.Room size: {roomsize}
5.Scene layout: 
{scene_layout}
6.Layout of door and windows"
{structure}
7.Rendered Image from the top view: SCENE_IMAGE.

**Note**:
"on" here means inside. "ontop" means on the supporter. Do not use "on" to represent the supporting relationship.
Here is your response, return a json format like the given example:
"""
