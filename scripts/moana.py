import json
import os
import struct
import sys
from pathlib import Path

import code

MoanaPath = Path(os.environ["MOANA_ROOT"]) / "island"
ScenePath = Path("../scene")

def corrected_transform(transform):
    # Moana column-major indices:
    #
    #  0  4  8 12
    #  1  5  9 13
    #  2  6 10 14
    #  3  7 11 15

    # Row-major, clipped indices
    return [
        transform[0], transform[4], transform[8], transform[12],
        transform[1], transform[5], transform[9], transform[13],
        transform[2], transform[6], transform[10], transform[14],
    ]

def write_transforms(filename, transforms):
    print("Writing:", filename)
    print("  Transform count:", len(transforms))
    output_file = open(filename, "wb")

    count_bin = struct.pack("i", len(transforms))
    output_file.write(count_bin)

    for transform in transforms:
        transform_bin = struct.pack("12f", *transform)
        output_file.write(transform_bin)

def find_all_archives(element_digest):
    archive_obj_files = set()

    instanced_copies = element_digest.get("instancedCopies", {}).values()
    instanced_primitives_by_copy = [
        # .values() discards name, eg "xgCabbage"
        search_root["instancedPrimitiveJsonFiles"].values()
        for search_root
        in [ element_digest ] + list(instanced_copies)
        if "instancedPrimitiveJsonFiles" in search_root
    ]

    for instanced_primitives in instanced_primitives_by_copy:
        archive_primitives_in_copy = [
            instanced_primitive
            for instanced_primitive
            in instanced_primitives
            if instanced_primitive["type"] == "archive"
        ]

        for archive_primitives in archive_primitives_in_copy:
            archive_obj_files.update(archive_primitives["archives"])

    return sorted(archive_obj_files)

def find_archive_digest_filenames(instanced_primitives_digest):
    return [
        instanced_primitives["jsonFile"]
        for instanced_primitives
        in instanced_primitives_digest.values()
        if instanced_primitives["type"] == "archive"
    ]

def process_archive_digest(digest_filename, copy_info):
    print(f"Processing archive digest: {digest_filename}")

    archive_digest = json.load(open(MoanaPath / digest_filename))

    for obj_filename, transform_items in archive_digest.items():
        transforms = transform_items.values()

        digest_stem = Path(digest_filename).stem
        obj_stem = Path(obj_filename).stem
        bin_filename = ScenePath / f"{digest_stem}--{obj_stem}.bin"

        write_transforms(
            bin_filename,
            [ corrected_transform(t) for t in transforms ]
        )
        copy_info.transform_bins.append((obj_filename, bin_filename))

class ElementCopyInfo:
    def __init__(self, transform):
        self.transforms = [ transform ]
        self.transform_bins = []

def process(element_name, output_cpp=False):
    print(f"Processing: {element_name}")

    element_path = f"json/{element_name}/{element_name}.json"

    element_digest = json.load(open(MoanaPath / element_path))
    root_geom_file = element_digest["geomObjFile"]
    element_copy_infos = {
        root_geom_file: ElementCopyInfo(element_digest["transformMatrix"])
    }

    instanced_primitives_digest = element_digest.get("instancedPrimitiveJsonFiles", {})
    archive_digest_filenames = find_archive_digest_filenames(instanced_primitives_digest)
    for archive_digest_filename in archive_digest_filenames:
        process_archive_digest(
            archive_digest_filename,
            element_copy_infos[root_geom_file]
        )

    instanced_copies = element_digest.get("instancedCopies", {}).values()


    for instanced_copy in instanced_copies:
        transform_matrix = instanced_copy["transformMatrix"]
        geom_obj_file = instanced_copy.get("geomObjFile", None)
        if geom_obj_file:
            assert geom_obj_file not in element_copy_infos
            element_copy_infos[geom_obj_file] = ElementCopyInfo(transform_matrix)
            current_copy_info = element_copy_infos[geom_obj_file]
        else:
            element_copy_infos[root_geom_file].transforms.append(transform_matrix)
            current_copy_info = element_copy_infos[root_geom_file]

        instanced_primitives_digest = instanced_copy.get("instancedPrimitiveJsonFiles", {})
        archive_digest_filenames = find_archive_digest_filenames(instanced_primitives_digest)
        for archive_digest_filename in archive_digest_filenames:
            process_archive_digest(
                archive_digest_filename,
                current_copy_info
            )

    obj_archives = find_all_archives(element_digest)

    base_obj_paths = []
    element_instances_bin_paths = []
    primitive_instances_bin_paths = []
    primitive_instances_handle_indices = []
    for geom_obj_file, copy_info in element_copy_infos.items():
        obj_stem = Path(geom_obj_file).stem
        bin_path = ScenePath / f"{obj_stem}.bin"
        write_transforms(
            bin_path,
            [ corrected_transform(t) for t in copy_info.transforms ]
        )

        base_obj_paths.append(geom_obj_file)
        element_instances_bin_paths.append(bin_path)

        primitive_instance_bin_paths = [
            bin_path
            for _, bin_path
            in copy_info.transform_bins
        ]
        primitive_instances_bin_paths.append(primitive_instance_bin_paths)

        primitive_instance_handle_indices = [
            obj_archives.index(obj_path)
            for obj_path, _
            in copy_info.transform_bins
        ]
        primitive_instances_handle_indices.append(primitive_instance_handle_indices)

    if output_cpp:
        print("Writing code:")
        code_dict = code.generate_code(
            element_name,
            base_obj_paths,
            element_instances_bin_paths,
            obj_archives,
            primitive_instances_bin_paths,
            primitive_instances_handle_indices
        )
        for filename, code_str in code_dict.items():
            code_path = Path("../src/") / filename
            print(f"  {code_path}")

            f = open(code_path, "w")
            f.write(code_str)

elements = [
    "isBayCedarA1",
    # "isBeach",
    "isCoastline",
    "isCoral",
    "isDunesA",
    "isDunesB",
    "isGardeniaA",
    "isHibiscus",
    "isHibiscusYoung",
    "isIronwoodA1",
    "isIronwoodB",
    "isKava",
    "isLavaRocks",
    "isMountainA",
    "isMountainB",
    "isNaupakaA",
    "isPalmDead",
    "isPalmRig",
    "isPandanusA",
    "osOcean",
]

def run():
    for element in elements:
        process(element, output_cpp=True)

def list_element_jsons():
    for element_name in elements:
        json_path = MoanaPath / f"json/{element_name}/{element_name}.json"
        print(json_path)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[-1] == "--list":
            list_element_jsons()
        else:
            print("Unknown argument:", sys.argv[-1])
            exit(1)
    else:
        run()
