import bpy
import os

# Folder Settings
input_folder = r"/path/to/folder/from/fbx"  # Change to the appropriate path
output_folder = r"/path/to/folder/to/obj"  # Change to the appropriate path

# Checking if folders exist
if not os.path.exists(input_folder):
    raise FileNotFoundError(f"The input folder does not exist: {input_folder}")
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to remove all objects in the scene
def clear_scene():
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

# Function to export objects to OBJ
def export_to_obj(filepath):
    # Save all selected objects to OBJ format
    with open(filepath, 'w') as obj_file:
        for obj in bpy.context.selected_objects:
            if obj.type == 'MESH':
                # Add object header
                obj_file.write(f'o {obj.name}\n')
                
                # Add vertices
                for vertex in obj.data.vertices:
                    co = obj.matrix_world @ vertex.co  # World coordinates of the vertices
                    obj_file.write(f'v {co.x} {co.y} {co.z}\n')

                # Add faces
                for poly in obj.data.polygons:
                    indices = [str(index + 1) for index in poly.vertices]  # Indexes start from 1 in OBJ
                    obj_file.write(f'f {" ".join(indices)}\n')

# Iterating through files in a folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".fbx"):
        # Full path to FBX file
        fbx_path = os.path.join(input_folder, filename)
        
        try:
            # Import FBX
            bpy.ops.import_scene.fbx(filepath=fbx_path)
            
            # Export settings
            obj_filename = os.path.splitext(filename)[0] + ".obj"
            obj_path = os.path.join(output_folder, obj_filename)

            # Export to OBJ
            export_to_obj(obj_path)

        except RuntimeError as e:
            print(f"Error while importing {filename}: {e}")

        # Cleaning the scene before the next import
        clear_scene()

print("Export completed!")
