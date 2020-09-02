import collections
import itertools
from pathlib import Path

import code
import files
import hardcoded_data
import materials
import params

TextureOffset = collections.namedtuple("TextureOffset", [
    "start", "end", "texture_index", "name"
])

class MeshRecord:
    def __init__(self, name, material, mtl_offset, element_name, is_texture, obj_file):
        self.name = name
        self.material = material
        self.mtl_offset = mtl_offset
        self.is_texture = is_texture
        self.element_name = element_name
        self.obj_file = obj_file

        self.primitive_count = 0

    def __repr__(self):
        return f"{{MeshRecord name={self.name} material={self.material} mtl_offset={self.mtl_offset} primitive_count={self.primitive_count} is_texture={self.is_texture} obj_file={self.obj_file} element_name={self.element_name}}}"

def parse_mesh_records(element_name, obj_filename, ptx_lookup):
    current_group = None
    current_mesh_record = None
    mtl_counts = collections.defaultdict(int)

    mesh_records = []
    with open(obj_filename, "r") as obj:
        for line in obj:
            tokens = line.strip().split()
            if not tokens: continue
            if tokens[0] == "g":
                if current_mesh_record:
                    mesh_records.append(current_mesh_record)
                    current_mesh_record = None

                group = tokens[1]
                if group == "default":
                    current_group = None
                else:
                    current_group = group

            if tokens[0] == "usemtl":
                try:
                    material = tokens[1]
                except IndexError:
                    if obj_filename.stem == "isCoral":
                        material = "coral"
                    else:
                        # fixme: really what we want?
                        material = "hidden"

                mesh_name = current_group
                current_mesh_record = MeshRecord(
                    mesh_name,
                    material,
                    mtl_counts[material],
                    element_name,
                    ptx_lookup.has_texture(element_name, mesh_name),
                    obj_filename.relative_to(params.MoanaPath)
                )
            if tokens[0] == "f":
                if current_mesh_record:
                    current_mesh_record.primitive_count += 2
                    mtl_counts[current_mesh_record.material] += 2
                else:
                    # fixme; make sure we're in authorized files
                    pass

        if current_mesh_record:
            mesh_records.append(current_mesh_record)
        else:
            # fixme; make sure we're in authorized files
            pass

        return mesh_records

def build_offsets(mesh_records, ptx_lookup, material_list):
    offsets_by_material_index = [ [] for _ in material_list ]

    for mesh_record in mesh_records:
        if not mesh_record.is_texture: continue

        try:
            key = (mesh_record.element_name, mesh_record.material)
            material_index = material_list.index(key)
        except ValueError:
            # fixme: use hardcoded_data
            if key == ("isDunesB", "isIronwood_archive_bark"):
                material_index = material_list.index(("isDunesB", "isIronwoodA_archive_bark"))

        primitive_offset = hardcoded_data.primitive_index_offsets.get(
            str(mesh_record.obj_file),
            0
        )

        offset = TextureOffset(
            mesh_record.mtl_offset + primitive_offset,
            mesh_record.mtl_offset + primitive_offset + mesh_record.primitive_count,
            ptx_lookup.index_of_stem(mesh_record.element_name, mesh_record.name),
            mesh_record.name
        )
        offsets_by_material_index[material_index].append(offset)

    for index in range(len(offsets_by_material_index)):
        offsets_by_material_index[index] = sorted(
            set(offsets_by_material_index[index]),
            key=lambda o: o.start
        )

    return offsets_by_material_index

def process_texture_offsets(elements):
    ptx_lookup = files.PtexLookup(params.elements)
    sbt_manager = materials.build_sbt_manager(params.elements)
    material_list = sbt_manager.get_material_list()

    offsets_by_material_index = [ [] for _ in material_list ]

    for element_name in elements:
        print(f"Parsing mesh records for: {element_name}")

        mesh_records_for_element = []

        for obj_filename in files.find_obj_files(element_name):
            print(f"  Parsing .obj: {obj_filename}")
            mesh_records = parse_mesh_records(element_name, obj_filename, ptx_lookup)
            mesh_records_for_element.extend(mesh_records)

        element_offsets = build_offsets(
            mesh_records_for_element,
            ptx_lookup,
            material_list
        )

        for i, offsets in enumerate(element_offsets):
            offsets_by_material_index[i].extend(offsets)

    code_str = code.generate_texture_offets(offsets_by_material_index, ptx_lookup)

    code_path = Path("../src/scene/texture_offsets.cpp")
    with open(code_path, "w") as f:
        f.write(code_str)

