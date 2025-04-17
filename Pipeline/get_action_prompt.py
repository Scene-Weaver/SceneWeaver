

system_prompt = """
You are an expert in 3D scene generation and optimization. Your task is to iteratively design a scene to make it as realistic, accurate, and controllable as possible.

Given the user's prompt, carefully analyze the scene and choose the most effective method from a set of available options to improve it. 
Your goal is to select the method that will provide the most significant enhancement based on the current scene’s data, available resources, and constraints.
You **must not** pay attention to the style and texture.
"""


methods_prompt = """
Available Methods for 3D Scene Design/Modification:
We have four methods to design or modify a 3D scene, each with different strengths and weaknesses. The goal is to select the most effective method for the current iteration based on the user's prompt and the existing scene data.
None of the methods can modify or control the style. Obmit the requirement of style and texture. Focus on the semantic and layout.

Method 1: Load real2sim indoor scene data
  Data Prior: Yes
  Supported Room Types: Living room, dining room, bedroom, bathroom, office, and classroom.
  Use Case 1: Ideal for generating foundational layouts for common room types.
  Strengths: Provides a ready-made, realistic layout based on real-world data. Easy to implement.
  Weaknesses: Limited room type diversity compared to other methods.

Method 2: Scene synthesis by neural network
  Data Prior: Yes
  Model: Trained on the 3D Front indoor dataset.
  Supported Room Types: Living room, bedroom, and dining room.
  Use Case 1: Creates a foundational layout for common room types.
  Strengths: Fast and flexible, making it ideal for rapid scene prototyping. Offers diverse configurations for living room, bedroom, and dining room.
  Weaknesses: Less accurate compared to method 1 (real2sim) and other more data-driven methods.

Method 3: Image generation + 3D reconstruction
  Data Prior: 2D image prior
  Use Case 1: When there is nothing on the large furniture (such as a table, cabinet, and desk), use this method to add some small objects on it.
  Strengths: Excellent for adding a group of objects on the top of a large furniture.(e.g., enriching a tabletop).
  Weaknesses: Can not add objects inside a container, such as objects in the shelf. Weak in generating the entire scene or layout. Can not add objects on the wall, ground, or ceiling. 

Method 4: Generate/implement scene with GPT
  Data Prior: LLM prior
  Room Type Diversity: Not limited to any specific room types.
  Use Case 1: Design room types that are not supported by methods 1 or 2 (e.g., more niche or custom room types). 
  Use Case 2: Add large objects in the current scene.
  Use Case 3: Add small objects in/on the large furniture, especially when method 3 does not work. (e.g., add a cup on the table or a book in the shelf).
  Strengths: Highly versatile and capable of generating scenes for any room type. Flexible with respect to room design and customization. Also expert in adding a single object 
  Weaknesses: May not be as real or accurate as data-driven methods. Can not modify objects in the current scene. Can not add objects on the wall, ground, or ceiling. 

Method 5: Modify layout with GPT
  Data Prior: LLM prior
  Room Type Diversity: Works with all room types.
  Use Case 1: When the layout has collision, out of room, or inproper placement, use this method to refine layouts by adjusting object placement. (e.g., reposition a chair)
  Use Case 2: When some object is redundant, use this method to remove redundant objects. (e.g., eliminate a table in the corner)
  Use Case 3: Add relations to some objects to make the room tidy.
  Strengths: Highly flexible and adaptable to various room designs. Excels at modifying or removing specific objects.
  Weaknesses: May lack precision and occasionally overlook details. Can not add objects.

Decision-Making Guide:
Based on the scene data, available methods, and specific requirements of the user’s prompt, choose the best method for the current iteration. The ideal method should balance speed, realism, controllability, and cost, depending on the specific problem you are aiming to solve.

If you need a foundational layout quickly, Method 1 offers the highest accuracy, while Method 2 provides greater flexibility and speed.
For unique or complex room types, Method 4 (Generated Scene with GPT) is ideal for creating a custom layout when other approaches fall short.
For detailed additions or partial scene enhancements, Method 4 (Generated Scene with GPT) are effective to add large furniture as well as implement details inside large furniture (such as add objects inside the shelf).
To add small objects on the large furniture's top (when there is nothing on it), use Method 3 (Image Generation + 3D Reconstruction) to add some small objects on it. (e.g., enriching a tabletop).
When the layout has collision, out of room, or inproper placement, use Method 5 (Modify layout with GPT) to refine layouts and add relations.
When some object is redundant, use Method 5 to remove redundant objects.

To achieve the best results, combine multiple methods over several iterations — start with a foundational layout and refine it progressively with finer details.

"""

# Method 5: Modify by rule
#   Data Prior: None
#   Use Case: Suitable for fine-tuning or making detailed modifications to specific objects within an existing scene.
#   Capabilities: Can add, remove, or move objects. Allows for modifying object relationships (positioning, constraints, etc.).
#   Strengths: Highly controllable for adjusting the scene layout and objects. Useful for precise, targeted edits. Also useful for adding objects inside a furniture randomly, such as adding books in the shelf.
#   Weaknesses: Not suitable for creating the scene from scratch or generating a foundational layout.


# sceneinfo_prompt = """
# The current layout of the scene is {scene_layout}.
# The image rendered from the top view is {scene_image}.
# The evaluated score is {eval_score}.
# "None" means the scene is empty.
# """

sceneinfo_prompt = """
Layout: {scene_layout}.
The Layout include each object's X-Y-Z Position, Z rotation, size (x_dim, y_dim, z_dim), as well as relation info with parents.

A rendered image of the entire scene taken from the top view: SCENE_IMAGE.
"None" means the scene is empty.

**3D Convention:**
- Right-handed coordinate system.
- The X-Y plane is the floor; the Z axis points up. The origin is at a corner (the left-top corner of the rendered image), defining the global frame.
- Asset front faces point along the positive X axis. The Z axis points up. The local origin is centered in X-Y and at the bottom in Z. 
A 90-degree Z rotation means that the object will face the positive Y axis. The bounding box aligns with the assets local frame.

"""

feedback_reflections_prompt = """
You are an experienced layout designer to design a 3D scene. 
Your goal is to match the given open-vocabulary category name with the standard category name.


You will receive:
1. The roomtype you need to design.
2. A list of given open-vocabulary category names.

You need to return a dict including:
1. The mapping of given category name with the most similar standard category name. 

*** Important ***
The standard category list: ['appliances.BeverageFridgeFactory', 'appliances.DishwasherFactory', 'appliances.MicrowaveFactory', 'appliances.OvenFactory', 'appliances.MonitorFactory', 'appliances.TVFactory', 'bathroom.BathroomSinkFactory', 'bathroom.StandingSinkFactory', 'bathroom.BathtubFactory', 'bathroom.HardwareFactory', 'bathroom.ToiletFactory', 'decor.AquariumTankFactory', 'elements.DoorCasingFactory', 'elements.GlassPanelDoorFactory', 'elements.LiteDoorFactory', 'elements.LouverDoorFactory', 'elements.PanelDoorFactory', 'elements.NatureShelfTrinketsFactory', 'elements.PillarFactory', 'elements.RugFactory', 'elements.CantileverStaircaseFactory', 'elements.CurvedStaircaseFactory', 'elements.LShapedStaircaseFactory', 'elements.SpiralStaircaseFactory', 'elements.StraightStaircaseFactory', 'elements.UShapedStaircaseFactory', 'elements.PalletFactory', 'elements.RackFactory', 'lamp.DeskLampFactory', 'lamp.FloorLampFactory', 'lamp.LampFactory', 'seating.BedFactory', 'seating.BedFrameFactory', 'seating.BarChairFactory', 'seating.ChairFactory', 'seating.OfficeChairFactory', 'seating.MattressFactory', 'seating.PillowFactory', 'seating.ArmChairFactory', 'seating.SofaFactory', 'shelves.CellShelfFactory', 'shelves.TVStandFactory', 'shelves.CountertopFactory', 'shelves.KitchenCabinetFactory', 'shelves.KitchenIslandFactory', 'shelves.KitchenSpaceFactory', 'shelves.LargeShelfFactory', 'shelves.SimpleBookcaseFactory', 'shelves.SidetableDeskFactory', 'shelves.SimpleDeskFactory', 'shelves.SingleCabinetFactory', 'shelves.TriangleShelfFactory', 'table_decorations.BookColumnFactory', 'table_decorations.BookStackFactory', 'table_decorations.SinkFactory', 'table_decorations.TapFactory', 'table_decorations.VaseFactory', 'tables.TableCocktailFactory', 'tables.CoffeeTableFactory', 'tables.SideTableFactory', 'tables.TableDiningFactory', 'tables.TableTopFactory', 'tableware.BottleFactory', 'tableware.BowlFactory', 'tableware.CanFactory', 'tableware.ChopsticksFactory', 'tableware.CupFactory', 'tableware.FoodBagFactory', 'tableware.FoodBoxFactory', 'tableware.ForkFactory', 'tableware.SpatulaFactory', 'tableware.FruitContainerFactory', 'tableware.JarFactory', 'tableware.KnifeFactory', 'tableware.LidFactory', 'tableware.PanFactory', 'tableware.LargePlantContainerFactory', 'tableware.PlantContainerFactory', 'tableware.PlateFactory', 'tableware.PotFactory', 'tableware.SpoonFactory', 'tableware.WineglassFactory', 'wall_decorations.BalloonFactory', 'wall_decorations.RangeHoodFactory', 'wall_decorations.MirrorFactory', 'wall_decorations.WallArtFactory', 'wall_decorations.WallShelfFactory']
You can only use category name from the standard list. If no standard category is matched, return null.

Here is the example: 
{
    "Roomtype": "{roomtype}"
    "list of given category names": ["bed", "nightstand", "lamp", "wardrobe"]
    "Mapping results": {"bed": "seating.BedFactory","nightstand": "shelves.SingleCabinetFactory","lamp": "lamp.DeskLampFactory", "wardrobe": null}
}
"""

feedback_reflections_prompt_system = """
{system_prompt}
{methods_prompt}
"""


# Previous Guide/Action:
# {previous_guide}
# The feedback shows results of previous action.
# If previous step failed in adding some objects, it means the attempt is invalid. Do not try again.


feedback_reflections_prompt_user = """
User’s Prompt: {user_demand}
RoomType: {roomtype}
Current Iteration: {iter}

Previous method, ideas and evaluated score:
{previous_guide}
If you've tried the same method several times and it's still not working, it's a sign that this approach isn't suitable for solving the problem. 
It's time to try shift your attention to another issue.

Current Scene:
{sceneinfo_prompt}
Describe the scene based on the provided image and the current data.

Analyze the scene and propose some improvements. You can consider the following factors:

Initial Scene Construction: If the scene is empty or lacks a foundational structure, what is the best method to generate a basic layout that fits the user's prompt (e.g., living room, office)?
Adding Large Furniture: If the scene feels sparse, assess whether more furniture needs to be added to the ground or corners.
Adding Small Objects: Add some small objects (such as in shelves, on desks, or on tables) to enrich the details. But do not make it crowded.
Detailing: Evaluate if the scene could benefit from small decorative items or functional objects placed inside or on top of furniture (e.g., books, lamps, plants).
Collision and Layout Issues: Check if there are any collisions or improper placements of objects that disrupt the flow of the room. Do any objects need repositioning or resizing for better usability or aesthetic?
User Prompt Satisfaction: Does the current scene meet the user's prompt requirements? What needs to be changed or added to align with the prompt more closely?
Realism Enhancement: What adjustments can be made to make the scene feel more realistic? Consider adding, removing, or repositioning objects to enhance visual harmony and authenticity.
Check Object: Check for any redundant or unnecessary objects that could be removed to streamline the scene.

Note: Do not add rugs on the ground or decorations on the wall. You **must not** pay attention to the style and texture.
You should use "remove" and "add" in two iterations rather than "replace" in one iteration, since it contains two steps.
Make your choice based on which method will achieve the highest quality and efficiency in improving the scene while keeping the user's needs and constraints in mind.

Main Goal for the Next Iteration:
Identify the most crucial improvement that should be made in this iteration based on the Analysis. This should address the most pressing issue and make the biggest impact on the overall scene.

You may update the scene over multiple iterations, and you can select the most appropriate method (only one) for each iteration. It’s often useful to leverage different methods in various iterations to make incremental improvements.

Recommended Approach & Method(s) for 5 Iterations:
You can consider using a sequence of methods that build on each other in multiple iterations. A sample approach could be:

Iteration 0: Use Method 1 or 2 to create a basic layout for the given scene type (if supported).
Iteration 1: Enhance the scene with Method 3 or 4, adding more details or making improvements based on the initial layout.
Iteration 2: Add more refinement and smaller details in specific areas (e.g., corners, objects on tables) using Method 3 or 4.
Iteration 3: Make fine-tuned modifications to specific objects using Method 5, focusing on layout adjustments and object removal.
Iteration 4: Everything is fine. Finish.
Try to avoid repeating the same method or actions in consecutive iterations.


Be as specific and concrete as possible, and write them out in a json format.
For example: 
{idea_example}

Your response:
"""
idea_example="""

{
  "iter":0,
  "Analysis of current scene": "Since there is no feedback provided, I will make an assessment based on the scene's prompt (bedroom) and the available resources and methods. At this stage, we're still in the first iteration, and the main goal is to create a foundational layout for the bedroom.",
  "Thoughts": "A solid and realistic starting point for the bedroom layout is crucial, so I should select a method that provides the most accurate and real representation of a bedroom. For this, Method 1 (real2sim indoor scene data) seems like the best option. This method can offer an accurate, data-driven layout of a typical bedroom, which will give us a solid base to work with.The other methods, such as Method 2 (scene synthesis by neural network), are also viable for providing a foundational layout, but Method 1 is a stronger choice due to its direct real-world data and higher realness.",
  "Recommendation": "For this iteration, I recommend using Method 1 (real2sim indoor scene data) to generate the initial bedroom layout. This will provide a reliable, realistic foundation for the scene, and in subsequent iterations, we can add more specific details or improve the layout using other methods. For example, we can add decorations or adjust the positioning of objects in the room in later iterations using Method 4 (Generated Scene Using GPT) or Method 3 (image generation + 3D reconstruction)."This approach ensures the scene starts with a solid and accurate layout, leaving room for refinements in future iterations.",
  "Method number": 1,
  "Ideas": "Add basic bedroom.",
  "RoomType": "bedroom"
}

{
  "iter":0,
  "Analysis of current scene": "In the first iteration, the goal is to create a foundational layout of the living room.",
  "Thoughts": "While Method 1 (real2sim indoor scene data) is a good choice, Method 2 (scene synthesis by neural network) can also provide a solid foundational layout. Although Method 1 is based on real-world data, Method 2 is quicker and may offer more flexibility with room layouts, especially if we want to explore different configurations for the living room. The model trained on the 3D Front indoor dataset provides reliable results for common room types, and this could allow for faster prototyping and adjustments.",
  "Recommendation": "For this iteration, I recommend using Method 2 (scene synthesis by neural network) to generate the initial living room layout. This method will allow us to quickly generate a realistic starting point for the room, providing a diverse and flexible foundation. In subsequent iterations, we can enhance the layout with additional details or refine object placements using other methods like Method 3 (image generation + 3D reconstruction) or Method 4 (Generated Scene Using GPT). This approach offers a balance of speed and flexibility while ensuring an accurate foundational layout.",
  "Method number":2,
  "Ideas": "Add basic livingroom.",
  "RoomType": "livingroom"
}
{
  "iter": 1,
  "Analysis of current scene": "The scene currently includes basic furniture such as a loveseat sofa, armchair, coffee table, bookshelves, and a TV stand. The layout appears functional but could benefit from additional detailing and decorative elements to enhance realism and aesthetic appeal.",
  "Thoughts": "The current layout serves as a good foundation, but it lacks personal touches and detailed elements that could make the space feel more lived-in and cozy. Adding decorative items like plants, books, and artwork, as well as functional items like lamps or a rug, could significantly improve the scene. Method 3 (image generation + 3D reconstruction) could be particularly useful for adding high-quality, realistic details and textures to the scene.",
  "Recommendation": "For this iteration, I recommend using Method 3 (image generation + 3D reconstruction) to add detailed and realistic decorative and functional items to the living room. This method will help in enhancing the visual appeal and realism of the scene by introducing elements that are typically found in a living room, such as decorative pillows, wall art, plants, and more textured materials for existing furniture. This approach will not only make the scene more visually appealing but also more aligned with what one might expect in a well-decorated living room.",
  "Method number": 3,
  "Ideas": "Add tissue and flower on the table.",
  "RoomType": "livingroom"
}
{
  "iter": 6,
  "Analysis of current scene": "The scene has been reviewed, and no immediate changes are necessary at this point. The overall layout and details are satisfactory.",
  "Thoughts": "The scene is functional and aesthetically pleasing as it currently stands. Further updates or additions may be unnecessary for the time being, as it already meets the design goals.",
  "Recommendation": "No further changes are recommended at this stage. The scene appears to be in good shape and does not require additional updates or modifications.",
  "Method number": 0,
  "Ideas": "No new ideas for this iteration.",
  "RoomType": "livingroom"
}
"""
