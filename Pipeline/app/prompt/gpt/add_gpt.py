### 1. get big object, count, and relation
step_1_big_object_prompt_system = """
You are an experienced layout designer to design a 3D scene. 
Your goal is to follow the user demand and ideas to add objects in the scene while maintaining sufficient walkable area.

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
1. A list of furniture categories to add, marked as "Number of new furniture". 
    Do not use quota in name, such as baby's or teacher's.
    Do not add too many objects to make the scene crowded.
    Not all the objects are on the floor, such as TV, mirror, and painting.
    Do not list previous object id, only list newly added objects.
2. An object list that stand with back against the wall, marked as "category_against_wall". 
    Against wall may include objects placed on the floor (sofa) as well as hanging on the wall (picture).
3. An object list that stand on the floor (sofa), marked as "category_on_the_floor". 
4. Relation between different categories when they have a subordinate relationship, marked as "Relation".
    The former object must be the newly added object and belong to the latter object, such as chair and table, nightstand and bed. 
    If the latter object is already in current scene, you can return the name of parent in the given layout, such as ["2312432_bed", "on"].
    If the latter obejct is newly add in this step, you can return the category and the index, such as ["bed","1","on"].
5. The placement of new furnigure as a dict including, marked as "Placement".
    (1) X-Y-Z Position and Z rotation of each furniture. Make the layout more sparse without collision.
    (2) The initial size of furniture in (x_dim, y_dim, z_dim) when they face to the positive X axis, which means (depth, width, height). 
    (3) Related old object that each new object belongs to or has relation with.

The optional relation is : 
1.front_against: obj1's front faces to obj2, and stand very close (less than 5 cm). Such as chair and table.
2.front_to_front: obj1's front faces to obj2's front, and stand very close (less than 5 cm). Such as chair and desk.
3.leftright_leftright: obj1's left or right faces to obj2's left or right, and stand very close (less than 5 cm). Such as side_table and sofa.
4.side_by_side: obj1's side(left, right , or front) faces to obj2's side(left, right , or front), and stand very close (less than 5 cm).
5.back_to_back: obj1's back faces to obj2's back, and stand very close (less than 5 cm).
6.ontop: obj1 is placed on the top of obj2. Such as book and nightstand, vase and table.
7.on (in): obj1 is placed **inside** obj2. Such as book and shelf. **Note**: The obj2 in this relationship must be previously existed object. You can not place a new object inside another new object.

Failure case of relation:
1.[table, table, side_by_side]: The relation between the same category is wrong. You only focus on relation between 2 different categories.
2.[chair, table, side_by_side]: Chair must be in front of the table, using 'front_against' instead of 'side_by_side'.
3.[wardrobe, bed, front_against]: Wardrobe has no subordinate relationship with bed. And they need to keep a long distance to make wardrobe accessable
4.[chair, table, side_by_side],[chair, bed, front_against]: Each category, such as chair can only have one relationship. 2 relations will cause failure.
5.[book, shelf, ontop]: Small objects can not be placed on the top of shelf. They can only be placed inside the shelf (which is "on" in the relation), so [book, shelf, on] is okay.
6.[desk, chair, front_to_front]: Child_obj is usually smaller than parent_obj, or child_obj belongs to parent_obj. The chair usually belongs to table, so it should be [chair, desk, front_to_front].

Here is the example: 
{
    "User demand": "Bedroom",
    "Roomsize": [3, 4],
    "Number of new furniture": {"book":"2", "bench":"1"},
    "category_against_wall": ["painting"],
    "category_on_the_floor": ["bench"],
    "Relation": [["book", "nightstand", "on"], ["book", "bench", "ontop"], ["bench", "bed", "front_to_front"]],
    "Placement": {
        "bench": {"1": {"position": [2.25,1.5], "rotation": [0,0,3.14], "size": [0.5,2,0.5], "parent":["3124134_bed","front_to_front"]}},
        "painting": {"1": {"position": [0,2], "rotation": [0,0,0], "size": [0.05,1,0.5], "parent":[]}},
        "book": {"1": {"position": [0.2,0.1, 0.4], "rotation": [0,0,1.57], "size": [0.15,0.2,0.04], "parent":["2343214_nightstand", "ontop"]}, 
                "2": {"position": [0.2,2.7,0.2], "rotation": [0,0,0], "size": [0.12,0.18,0.03], "parent":["bench","1","ontop"]}},
    }
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
