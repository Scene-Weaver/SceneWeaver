### 1. get big object, count, and relation
step_1_big_object_prompt_system = """
You are an experienced layout designer to design a 3D scene. 
Your goal is to help me choose objects to put in the scene.

You will receive:
1. The roomtype you need to design.

You need to return a dict including:
1. Room size, including width and height in meters.
2. A list of big-furniture categories that stand on the floor, marked with count. 
You can refer but not limited to this category list: ['BeverageFridge', 'Dishwasher', 'Microwave', 'Oven', 'Monitor', 'TV', 'BathroomSink', 'StandingSink', 'Bathtub', 'Hardware', 'Toilet', 'AquariumTank', 'DoorCasing', 'GlassPanelDoor', 'LiteDoor', 'LouverDoor', 'PanelDoor', 'NatureShelfTrinkets', 'Pillar', 'CantileverStaircase', 'CurvedStaircase', 'LShapedStaircase', 'SpiralStaircase', 'StraightStaircase', 'UShapedStaircase', 'Pallet', 'Rack', 'CeilingClassicLamp', 'CeilingLight', 'DeskLamp', 'FloorLamp', 'Lamp', 'Bed', 'BedFrame', 'BarChair', 'Chair', 'OfficeChair', 'Mattress', 'Pillow', 'ArmChair', 'Sofa', 'CellShelf', 'TVStand', 'Countertop', 'KitchenCabinet', 'KitchenIsland', 'KitchenSpace', 'LargeShelf', 'SimpleBookcase', 'SidetableDesk', 'SimpleDesk', 'SingleCabinet', 'TriangleShelf', 'BookColumn', 'BookStack', 'Sink', 'Tap', 'Vase', 'TableCocktail', 'CoffeeTable', 'SideTable', 'TableDining', 'TableTop', 'Bottle', 'Bowl', 'Can', 'Chopsticks', 'Cup', 'FoodBag', 'FoodBox', 'Fork', 'Spatula', 'FruitContainer', 'Jar', 'Knife', 'Lid', 'Pan', 'LargePlantContainer', 'PlantContainer', 'Plate', 'Pot', 'Spoon', 'Wineglass', 'Balloon', 'RangeHood', 'Mirror', 'WallArt', 'WallShelf']
Do not return objects like rug. 
Do not use quota in name, such as baby's or teacher's.
Enhance the immersion of the scene by incorporating more categories and increasing their quantities.
The total count of furniture must be larger than 15.
3. An object list that stand with back against the wall
4. Relation between different categories when they have a subordinate relationship and stay very close(less than 5 cm).
The former object is smaller than the latter object, such as chair and table, nightstand and bed. 


The optional relation is : 
1.front_against: obj1's front faces to obj2, and stand very close.
2.front_to_front: obj1's front faces to obj2's front, and stand very close.
3.leftright_leftright: obj1's left or right faces to obj2's left or right, and stand very close. 
4.side_by_side: obj1's side(left, right , or front) faces to obj2's side(left, right , or front), and stand very close. 
5.back_to_back: obj1's back faces to obj2's back, and stand very close. 

Failure case of relation:
1.[table, table, side_by_side]: The relation between the same category is wrong. You only focus on relation between 2 different categories.
2.[chair, table, side_by_side]: Chair must be in front of the table, using 'front_against' instead of 'side_by_side'.
3.[wardrobe, bed, front_against]: Wardrobe has no subordinate relationship with bed. And they need to keep a long distance to make wardrobe accessable
4.[chair, table, side_by_side],[chair, bed, front_against]: Each category, such as chair can only have one relationship. 2 relations will cause failure.

Here is the example: 
{
    "Roomtype": "Bedroom",
    "Size": [3, 4],
    "Category list of big object": {"bed":"1", "wardrobe":"1", "nightstand":"2", "bench":"1"},
    "Object against the wall": ["bed", "wardrobe", "nightstand"],
    "Relation between big objects": [["nightstand", "bed", "side_by_side"], ["bench", "bed", "front_to_front"]]
}
"""
step_1_big_object_prompt_user = """
Here is the roomtype you need to design:
Roomtype: {roomtype}

Here is your response:
"""



#### 2. get small object and relation

step_2_small_object_prompt_system = """
You are an experienced layout designer to design a 3D scene. 
Your goal is to help me choose small objects to put in the scene.

You will receive:
1. The roomtype you need to design.
2. The big furniture that exist in this room.

You need to return a dict including:
1. A list of small-furniture categories that belongs to (on or inside) the big furniture. Format as a python list. 
Use [book] instead of ["book"]. Do not use quota in name, such as baby's or teacher's.
Enhance the immersion of the scene by incorporating more categories and increasing their quantities.
2. Relation between small furniture and big furniture, with count for each big furniture.
The former object is smaller than the latter object, such as laptop and desk. 

The optional relation is : 
1.ontop: obj1 is placed on the top of obj2.
2.on: obj1 is placed on the top of or inside obj2.

Here is the example: 
{
    "Roomtype": "Bedroom",
    "List of big furniture": ["bed", "wardrobe", "nightstand", "bench"],
    "List of small furniture": ["book", "plant", "lamp", "clothes"],
    "Relation": ["book", "nightstand", "on", "2"], ["plant", "nightstand", "ontop", "1"], ["lamp", "nightstand", "ontop", "1"], ["clothes", "bench", "ontop", "2"], ["clothes", "wardrobe", "on", "4"]
}

"""

step_2_small_object_prompt_user = """
Here is the roomtype you need to design:
Roomtype: {roomtype}
List of big furniture: {big_category_list}

Here is your response:
"""


#### 3. get object class name in infinigen

step_3_class_name_prompt_system = """
You are an experienced layout designer to design a 3D scene. 
Your goal is to match the given open-vocabulary category name with the standard category name.


You will receive:
1. The roomtype you need to design.
2. A list of given open-vocabulary category names.

You need to return a dict including:
1. The mapping of given category name with the most similar standard category name. 

*** Important ***
The standard category list: ['appliances.BeverageFridgeFactory', 'appliances.DishwasherFactory', 'appliances.MicrowaveFactory', 'appliances.OvenFactory', 'appliances.MonitorFactory', 'appliances.TVFactory', 'bathroom.BathroomSinkFactory', 'bathroom.StandingSinkFactory', 'bathroom.BathtubFactory', 'bathroom.HardwareFactory', 'bathroom.ToiletFactory', 'decor.AquariumTankFactory', 'elements.DoorCasingFactory', 'elements.GlassPanelDoorFactory', 'elements.LiteDoorFactory', 'elements.LouverDoorFactory', 'elements.PanelDoorFactory', 'elements.NatureShelfTrinketsFactory', 'elements.PillarFactory', 'elements.RugFactory', 'elements.CantileverStaircaseFactory', 'elements.CurvedStaircaseFactory', 'elements.LShapedStaircaseFactory', 'elements.SpiralStaircaseFactory', 'elements.StraightStaircaseFactory', 'elements.UShapedStaircaseFactory', 'elements.PalletFactory', 'elements.RackFactory', 'lamp.CeilingClassicLampFactory', 'lamp.CeilingLightFactory', 'lamp.DeskLampFactory', 'lamp.FloorLampFactory', 'lamp.LampFactory', 'seating.BedFactory', 'seating.BedFrameFactory', 'seating.BarChairFactory', 'seating.ChairFactory', 'seating.OfficeChairFactory', 'seating.MattressFactory', 'seating.PillowFactory', 'seating.ArmChairFactory', 'seating.SofaFactory', 'shelves.CellShelfFactory', 'shelves.TVStandFactory', 'shelves.CountertopFactory', 'shelves.KitchenCabinetFactory', 'shelves.KitchenIslandFactory', 'shelves.KitchenSpaceFactory', 'shelves.LargeShelfFactory', 'shelves.SimpleBookcaseFactory', 'shelves.SidetableDeskFactory', 'shelves.SimpleDeskFactory', 'shelves.SingleCabinetFactory', 'shelves.TriangleShelfFactory', 'table_decorations.BookColumnFactory', 'table_decorations.BookStackFactory', 'table_decorations.SinkFactory', 'table_decorations.TapFactory', 'table_decorations.VaseFactory', 'tables.TableCocktailFactory', 'tables.CoffeeTableFactory', 'tables.SideTableFactory', 'tables.TableDiningFactory', 'tables.TableTopFactory', 'tableware.BottleFactory', 'tableware.BowlFactory', 'tableware.CanFactory', 'tableware.ChopsticksFactory', 'tableware.CupFactory', 'tableware.FoodBagFactory', 'tableware.FoodBoxFactory', 'tableware.ForkFactory', 'tableware.SpatulaFactory', 'tableware.FruitContainerFactory', 'tableware.JarFactory', 'tableware.KnifeFactory', 'tableware.LidFactory', 'tableware.PanFactory', 'tableware.LargePlantContainerFactory', 'tableware.PlantContainerFactory', 'tableware.PlateFactory', 'tableware.PotFactory', 'tableware.SpoonFactory', 'tableware.WineglassFactory', 'wall_decorations.BalloonFactory', 'wall_decorations.RangeHoodFactory', 'wall_decorations.MirrorFactory', 'wall_decorations.WallArtFactory', 'wall_decorations.WallShelfFactory']
You can only use category name from the standard list. If no standard category is matched, return null.

Here is the example: 
{
    "Roomtype": "{roomtype}"
    "list of given category names": ["bed", "nightstand", "lamp", "wardrobe"]
    "Mapping results": {"bed": "seating.BedFactory","nightstand": "shelves.SingleCabinetFactory","lamp": "lamp.DeskLampFactory", "wardrobe": null}
}
"""
step_3_class_name_prompt_user = """
Here is the roomtype you need to design:
Roomtype: Bookstore
List of given category names:  {category_list}

Here is your response:
"""

# ['BeverageFridgeFactory', 'DishwasherFactory', 'MicrowaveFactory', 'OvenFactory', 'MonitorFactory', 'TVFactory', 
#  'BathroomSinkFactory', 'StandingSinkFactory', 'BathtubFactory', 'HardwareFactory', 'ToiletFactory', 
#  'AquariumTankFactory', 'DoorCasingFactory', 'GlassPanelDoorFactory', 'LiteDoorFactory', 'LouverDoorFactory', 
#  'PanelDoorFactory', 'NatureShelfTrinketsFactory', 'PillarFactory', 'RugFactory', 'CantileverStaircaseFactory', 
#  'CurvedStaircaseFactory', 'LShapedStaircaseFactory', 'SpiralStaircaseFactory', 'StraightStaircaseFactory', 
#  'UShapedStaircaseFactory', 'PalletFactory', 'RackFactory', 'CeilingClassicLampFactory', 'CeilingLightFactory', 
#  'DeskLampFactory', 'FloorLampFactory', 'LampFactory', 'BedFactory', 'BedFrameFactory', 'BarChairFactory', 
#  'ChairFactory', 'OfficeChairFactory', 'MattressFactory', 'PillowFactory', 'ArmChairFactory', 'SofaFactory', 
#  'CellShelfFactory', 'TVStandFactory', 'CountertopFactory', 'CabinetDoorBaseFactory', 'KitchenCabinetFactory', 
#  'KitchenIslandFactory', 'KitchenSpaceFactory', 'LargeShelfFactory', 'SimpleBookcaseFactory', 'SidetableDeskFactory', 
#  'SimpleDeskFactory', 'SingleCabinetFactory', 'TriangleShelfFactory', 'BookColumnFactory', 'BookFactory', 
#  'BookStackFactory', 'SinkFactory', 'TapFactory', 'VaseFactory', 'TableCocktailFactory', 
#  'CoffeeTableFactory', 'SideTableFactory', 'TableDiningFactory', 'TableTopFactory', 'BottleFactory', 'BowlFactory', 
#  'CanFactory', 'ChopsticksFactory', 'CupFactory', 'FoodBagFactory', 'FoodBoxFactory', 'ForkFactory', 
#  'SpatulaFactory', 'FruitContainerFactory', 'JarFactory', 'KnifeFactory', 'LidFactory', 'PanFactory', 
#  'LargePlantContainerFactory', 'PlantContainerFactory', 'PlateFactory', 'PotFactory', 'SpoonFactory', 'WineglassFactory']

#### 4. generate rule code

step_4_rule_prompt_system = """
You are an experienced layout designer to design a 3D scene. 
Your goal is to write a python code to present the given designing rule.

You will receive:
1. The roomtype you need to design.
2. Rules to place the objects, including the relation between objets.
3. A partialy writen code with objects defined as variables in the scene.

You need to return:
1. The completed python code to present the given designing rules and relations.

* Note *
The relation should be writen as [cu.front_against, cu.front_to_front, cu.leftright_leftright, cu.side_by_side, cu.back_to_back] in python code.
If the relation is not written in the variable's definition, you should embed it in the constraints, such as the nightstand_obj in the following example.
You can not change the given variable's definition. You just need to write the constraints with the given variables.
Do not use functions that are not shown in the example. 

* Here is the example: *

Roomtype: Bedroom

Big-object count: 
{"bed":"1", "desks":"1", "floor lamps":"1", "nightstand":"2"}

Relation between big object:
[nightstand, beds, leftright_leftright]

Small object count and relation with big object:
[books, nightstand, on, 3]
[books, desks, ontop, 1]

* The variable's definition: *

rooms = cl.scene()[{Semantics.Room, -Semantics.Object}]
obj = cl.scene()[{Semantics.Object, -Semantics.Room}]
newroom = rooms[Semantics.NewRoom].excludes(cu.room_types)

constraints = OrderedDict()
score_terms = OrderedDict()

furniture = obj[Semantics.Furniture].related_to(rooms, cu.on_floor)
wallfurn = furniture.related_to(rooms, cu.against_wall)

beds_obj = wallfurn[seating.BedFactory]
desks_obj = wallfurn[shelves.SimpleDeskFactory]
nightstand_obj = wallfurn[shelves.SingleCabinetFactory]
floor_lamps_obj = obj[lamp.FloorLampFactory].related_to(rooms, cu.on_floor).related_to(rooms, cu.against_wall)
books_obj = obj[table_decorations.BookStackFactory]

* The constraints code: *

constraints["bedroom"] = newroom.all(
    lambda r: (
        beds_obj.related_to(r).count().in_range(1, 1)
        * (
            nightstand_obj.related_to(r)
            .related_to(beds_obj.related_to(r), cu.leftright_leftright)
            .count()
            .in_range(2, 2)
        )
        * desks_obj.related_to(r).count().in_range(1, 1)
        * desks_obj.related_to(r).all(
            lambda s: (
                books_obj.related_to(s, cu.ontop).count().in_range(1,1)
                * (books_obj.related_to(s, cu.ontop).count() >= 0)
            )
        )
        * floor_lamps_obj.related_to(r).count().in_range(1, 1)
        * nightstand_obj.related_to(r).all(
            lambda s: (
                books_obj.related_to(s, cu.on).count().in_range(3,3)
                * (books_obj.related_to(s, cu.on).count() >= 0)
            )
        )
    )
)


"""
step_4_rule_prompt_user = """
*Here is the roomtype and object info:*

Roomtype: {roomtype}

Big-object count: 
{big_category_cnt}

Relation between big object: 
{relation_big_object}

Small object count and relation with big object:
{relation_small_object}

* Here is the code you need to write constraints: *

rooms = cl.scene()[{{Semantics.Room, -Semantics.Object}}]
obj = cl.scene()[{{Semantics.Object, -Semantics.Room}}]
newroom = rooms[Semantics.Conference].excludes(cu.room_types)

constraints = OrderedDict()
score_terms = OrderedDict()

furniture = obj[Semantics.Furniture].related_to(rooms, cu.on_floor)
wallfurn = furniture.related_to(rooms, cu.against_wall)

{vars_definition}

* Here is your response of the constraints code: *
"""