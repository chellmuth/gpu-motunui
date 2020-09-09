import json
import os
from pathlib import Path

import hardcoded_data
from params import MoanaPath

class PtexLookup:
    def __init__(self, elements):
        self.lookup = []
        self.ptxs = []

        for element_name in elements:
            ptx_files = sorted(find_ptx_files(element_name, full_path=False))
            mapped_files = [
                (element_name, ptx_file.stem)
                for ptx_file
                in ptx_files
            ]

            self.lookup.extend(mapped_files)
            self.ptxs.extend(ptx_files)

    def has_texture(self, element_name, stem):
        if (element_name, stem) in self.lookup:
            return True

        if element_name in hardcoded_data.cross_element_texture_access:
            cross_element = hardcoded_data.cross_element_texture_access[element_name]
            return (cross_element, stem) in self.lookup

    def index_of_stem(self, element_name, stem):
        original_key = (element_name, stem)
        if original_key in self.lookup:
            return self.lookup.index(original_key)

        cross_element = hardcoded_data.cross_element_texture_access[element_name]
        cross_key = (cross_element, stem)
        return self.lookup.index(cross_key)

    def filename_list(self):
        return self.ptxs

def find_ptx_files(element_name, full_path=True):
    results = []

    root_path = MoanaPath / f"textures/{element_name}"
    for root, directories, filenames in os.walk(root_path):
        if "Displacement" in root: continue

        for filename in filenames:
            if filename.endswith(".ptx"):
                path = Path(root) / filename
                if full_path:
                    results.append(path)
                else:
                    results.append(path.relative_to(MoanaPath))

    return results

def find_obj_files(element_name):
    element_path = f"json/{element_name}/{element_name}.json"
    element_json = json.load(open(MoanaPath / element_path))

    roots = [element_json]
    roots.extend([
        instanced_copy
        for instanced_copy
        in element_json.get("instancedCopies", {}).values()
    ])

    results = set()
    for root in roots:
        key = "geomObjFile"
        if key in root:
            results.add(root[key])

        primitives = root.get("instancedPrimitiveJsonFiles", {})
        for primitives_item in primitives.values():
            if primitives_item["type"] == "archive":
                results.update(primitives_item["archives"])

    return [
        MoanaPath / obj_file for obj_file in sorted(results)
    ]

def find_material_json_file(element_name):
    element_path = f"json/{element_name}/{element_name}.json"
    element_json = json.load(open(MoanaPath / element_path))

    return MoanaPath / element_json["matFile"]
