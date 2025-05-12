system_prompt = """
I want to design a scene according to the user's demand. 
I have retrieved a scene from the dataset. 
I want to design the scene based on the retrieved scene. 
The size of the final scene should not be smaller than the retrieved scene. 
Help me design the room size.

You will reveive the following information:

1.User's demand of the scene
2.Roomtype: roomtype
3.Ideas about the scene
4.Current room size: length, width of the retrieved scene

Determine the final room size that best accommodates the user's demand while maintaining a coherent and functional design. 
The new dimensions (length, width) can be larger or smaller than the retrieved scene, ensuring a well-balanced layout that integrates all necessary elements effectively."

Example Input:
1.User's demand: A bedroom
2.Roomtype: bedroom
3.Ideas: Add basic bedroom
4.Current room size: 3.94,4.68

Expected Output: (do not explain, just return value)
4,5

"""
user_prompt = """
Input:
User's demand: {user_demand}
Roomtype: {roomtype}
Ideas: {ideas}
Current room size: {roomsize}

Your response:
"""