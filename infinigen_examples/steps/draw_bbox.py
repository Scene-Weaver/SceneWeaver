import bpy
from mathutils import Vector, Matrix

def add_rotated_bbox_wireframe(obj,cat_name):
    if obj.type != 'MESH':
        return

    # Create a cube with unit size at origin
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0))
    bbox_cube = bpy.context.active_object
    bbox_cube.name = f"RotatedBBOX_{obj.name}"
    print(bbox_cube.name)

    # Scale it to match the object's local bounding box
    local_bbox = [Vector(corner) for corner in obj.bound_box]
    min_corner = Vector((min(c[i] for c in local_bbox) for i in range(3)))
    max_corner = Vector((max(c[i] for c in local_bbox) for i in range(3)))
    size = max_corner - min_corner
    center = min_corner + size / 2

    bbox_cube.scale = size 
    scale_matrix = Matrix.Diagonal(size).to_4x4()
    bbox_cube.matrix_world = obj.matrix_world @ Matrix.Translation(center) @ scale_matrix

    # Add wireframe modifier
    mod = bbox_cube.modifiers.new("Wireframe", type='WIREFRAME')
    mod.thickness = 0.03

    mat = bpy.data.materials.new(name="WireframeMaterial")
    mat.use_nodes = True
    # Access the Principled BSDF node
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs['Base Color'].default_value = (0.0, 0.3, 1.0, 1.0)  # RGBA: blue
        bsdf.inputs['Specular'].default_value = 0  # Optional: make it less shiny
    bbox_cube.data.materials.append(mat)
    bbox_cube.display_type = 'WIRE'
    bbox_cube.hide_render = False
    
    add_text_on_bbox_surface(bbox_cube, text_content=cat_name)
    
    return



def add_text_on_bbox_surface(bbox_obj, text_content="ObjectName"):
    # Create text object
    bpy.ops.object.text_add(location=(0, 0, 0))
    text_obj = bpy.context.active_object
    text_obj.name = f"Label_{bbox_obj.name}"
    text_obj.data.body = text_content

    # Rotate and scale
    text_obj.rotation_euler = (0, 0, 0)
    scale = 0.15
    text_obj.scale = (scale, scale, scale)

    # Create material for white text
    text_mat = bpy.data.materials.new(name="TextWhite")
    text_mat.use_nodes = True
    bsdf = text_mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (1, 1, 1, 1)  # White
        bsdf.inputs["Emission"].default_value = (1, 1, 1, 1)    # Glow a bit
        bsdf.inputs["Emission Strength"].default_value = 2.0
    text_obj.data.materials.append(text_mat)

    # Position text above bbox
    bbox_center = bbox_obj.matrix_world.translation
    bbox_size = bbox_obj.dimensions
    text_offset = Vector((-text_obj.dimensions[0]*text_obj.scale[0]/2, 0.1, bbox_size.z / 2 + 0.02))
    text_obj.location = bbox_center + text_offset

    # Create background plane
#    bpy.ops.mesh.primitive_plane_add(size=1, location=text_obj.location - Vector((0, 0, 0.005)))
    bpy.ops.mesh.primitive_plane_add(size=1)
    bg_plane = bpy.context.active_object
    bg_plane.name = f"TextBackground_{bbox_obj.name}"
#    bg_plane.scale = (2+len(text_content)*0.34, 0.8, 1)
#    bg_plane.location = [0.8+0.17*len(text_content),0.3,-0.01]
    
    
    padding = 0.05
    bpy.context.view_layer.update()  
    text_size = text_obj.dimensions.xy
    bg_plane.scale.x = (text_size.x  + padding )/text_obj.scale[0]
    bg_plane.scale.y = (text_size.y  + padding) / text_obj.scale[1]
    bg_plane.location.x = (text_size.x / 2 )/ text_obj.scale[0]
    bg_plane.location.y = (text_size.y / 2 )/ text_obj.scale[1]
    bg_plane.location.z = -0.01
    # Match rotation if needed
    #    bg_plane.rotation_euler = text_obj.rotation_euler

    # Blue material for background
    bg_mat = bpy.data.materials.new(name="BG_Blue")
    bg_mat.use_nodes = True
    bg_bsdf = bg_mat.node_tree.nodes.get("Principled BSDF")
    if bg_bsdf:
        bg_bsdf.inputs["Base Color"].default_value = (0.0, 0.3, 1.0, 1.0)  # Blue
        bg_bsdf.inputs["Roughness"].default_value = 1.0
        
    bg_plane.data.materials.append(bg_mat)

    # Optional: parent text and background to bbox
#    text_obj.parent = bbox_obj
    bg_plane.parent = text_obj

    return text_obj, bg_plane

# #        
# #bbox_obj = bpy.data.objects.get("RotatedBBOX_MetaCategoryFactory(3629900).spawn_asset(9447620)")
# #add_text_on_bbox_surface(bbox_obj, text_content="ObjectName")