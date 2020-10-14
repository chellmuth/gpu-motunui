import collections
import json
from pathlib import Path

import code
import files
import params

def generate_texture_lookup_data(texture_index_lookup):
    items = "\n".join(
        f"X(\"{element_name}\", \"{material}\", \"{mesh}\", {index})"
        for (element_name, material, mesh), index
        in texture_index_lookup.items()
    )
    return items

def generate_texture_lookup_code():
    master_ptx_file_list = []
    for element_name in params.elements:
        ptx_files = files.find_ptx_files(element_name, full_path=False)
        master_ptx_file_list.extend(ptx_files)

    # Lots of false positives, includes all meshes for a given element
    # Could work backwards and build from meshes found in .obj files
    mesh_names_by_directory = collections.defaultdict(list)
    for ptx_file in master_ptx_file_list:
        mesh_name = ptx_file.stem
        mesh_names_by_directory[ptx_file.parent].append(mesh_name)

    texture_directory_lookup = {}
    for element_name in params.elements:
        materials_file = files.find_material_json_file(element_name)
        materials_json = json.load(open(materials_file, "r"))

        for material, material_json in materials_json.items():
            texture_directory = material_json.get("colorMap", "")
            key = (element_name, material)
            texture_directory_lookup[key] = Path(texture_directory)

    # fixme: move to overrides
    texture_directory_lookup[("isDunesB", "soilSimple")] = Path("textures/isDunesB/Color")

    texture_index_lookup = {}
    for (element, material), directory in texture_directory_lookup.items():
        available_mesh_names = mesh_names_by_directory[directory]
        for mesh_name in available_mesh_names:
            ptx_file = directory / f"{mesh_name}.ptx"
            ptx_index = master_ptx_file_list.index(ptx_file)
            texture_index_lookup[(element, material, mesh_name)] = ptx_index

    lookup_code = generate_texture_lookup_data(texture_index_lookup)
    with open("../src/scene/data/texture_lookup_data.cpp", "w") as f:
        f.write(lookup_code)

    filenames_code = code.generate_texture_filenames_data(master_ptx_file_list)
    with open("../src/scene/data/texture_offsets_data.cpp", "w") as f:
        f.write(filenames_code)

if __name__ == "__main__":
    generate_texture_lookup_code()
