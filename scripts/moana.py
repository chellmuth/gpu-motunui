import collections
import json
import struct
import sys
from pathlib import Path

import code
import curves
import materials
import textures
import transforms as transform_util
from params import MoanaPath, ScenePath, elements, skip_list

class ElementInstanceInfo:
    def __init__(self, transform):
        self.transforms = [ transform ]
        self.transform_bins = []
        self.curve_records = []

CurveInfo = collections.namedtuple(
    "CurveInfo",
    [
        "json_path",
        "width_root",
        "width_tip",
        "assignment_name",
    ]
)

CurveRecord = collections.namedtuple(
    "CurveRecord",
    [
        "assignment_name",
        "bin_path",
    ]
)

CurveDigestBlacklist = set([
    "json/isMountainB/isMountainB_xgLowGrowth.json"
])

# List of every archive found in the element json
#
# See: isNaupakaA.json, where xgBonsai_isNaupakaBon_bon_hero_ALL.obj is used by
# every instanced copy, because each copy has re-posed geometry, causing the
# primitives to need unique transforms (isNaupakaA1_xgBonsai.json).
def find_all_archives(element_json):
    archive_obj_files = set()

    instanced_copies = element_json.get("instancedCopies", {}).values()
    instanced_primitives_by_copy = [
        # .values() discards name, eg "xgCabbage"
        search_root["instancedPrimitiveJsonFiles"].values()
        for search_root
        in [ element_json ] + list(instanced_copies)
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

# List of every archive digest filename (eg isNaupakaA1_xgBonsai.json) found in
# a specific instancedPrimitiveJsonFiles section
#
# Used to build the list of files to process into transform bins.
def find_archive_digest_filenames(primitives_json):
    return [
        instanced_primitives["jsonFile"]
        for instanced_primitives
        in primitives_json.values()
        if instanced_primitives["type"] == "archive"
    ]

# List of CurveInfos for every curve section found in a specified
# instancedPrimitiveJsonFiles section
#
# Used to build the list of files to process into curve bins. CurveInfos are
# used instead of filenames because there is additional information in the
# curve section (root and tip widths)
def find_curve_infos(primitives_json):
    return [
        CurveInfo(
            MoanaPath / instanced_primitives["jsonFile"],
            instanced_primitives["widthRoot"],
            instanced_primitives["widthTip"],
            assignment_name
        )
        for assignment_name, instanced_primitives
        in primitives_json.items()
        if instanced_primitives["type"] == "curve"
        and instanced_primitives["jsonFile"] not in CurveDigestBlacklist
    ]

def process_archive_digest(digest_filename, element_instance_info):
    print(f"Processing archive digest: {digest_filename}")

    archive_digest = json.load(open(MoanaPath / digest_filename))

    for obj_filename, transform_items in archive_digest.items():
        transforms = transform_items.values()

        digest_stem = Path(digest_filename).stem
        obj_stem = Path(obj_filename).stem
        bin_filename = ScenePath / f"{digest_stem}--{obj_stem}.bin"

        transform_util.write_transforms(bin_filename, transforms)
        element_instance_info.transform_bins.append((obj_filename, bin_filename))

def process_curve_info(curve_info, element_instance_info):
    bin_path = ScenePath / f"curves__{curve_info.json_path.stem}.bin"
    curves.write_curve_bin(curve_info, bin_path)

    curve_record = CurveRecord(
        curve_info.assignment_name,
        bin_path,
    )

    element_instance_info.curve_records.append(curve_record)

def process_element_instance_json(instance_digest, instance_infos, root_geom_file):
    # Store the transform in table indexed by the instance's geometry
    transform_matrix = instance_digest["transformMatrix"]
    geom_obj_file = instance_digest.get("geomObjFile", None)
    if geom_obj_file:
        assert geom_obj_file not in instance_infos
        instance_infos[geom_obj_file] = ElementInstanceInfo(transform_matrix)
        current_instance_info = instance_infos[geom_obj_file]
    else:
        instance_infos[root_geom_file].transforms.append(transform_matrix)
        current_instance_info = instance_infos[root_geom_file]

    # Process instance's archives
    primitives_json = instance_digest.get("instancedPrimitiveJsonFiles", {})
    archive_digest_filenames = find_archive_digest_filenames(primitives_json)
    for archive_digest_filename in archive_digest_filenames:
        process_archive_digest(
            archive_digest_filename,
            current_instance_info
        )

    # Process instance's curves
    curve_infos = find_curve_infos(primitives_json)
    for curve_info in curve_infos:
        process_curve_info(
            curve_info,
            current_instance_info
        )

def process_element(element_name, sbt_manager, output_cpp=False):
    print(f"Processing geometry: {element_name}")

    element_path = f"json/{element_name}/{element_name}.json"
    element_json = json.load(open(MoanaPath / element_path))

    instance_infos = {}
    root_geom_file = element_json["geomObjFile"]

    # Gather info on the top level element instance
    process_element_instance_json(
        element_json,
        instance_infos,
        root_geom_file
    )

    # Gather info in the instancedCopies section
    instanced_copies = element_json.get("instancedCopies", {}).values()
    for instanced_copy in instanced_copies:
        process_element_instance_json(
            instanced_copy,
            instance_infos,
            root_geom_file
        )

    # Write the gathered element instance transforms to disk
    for geom_obj_file, copy_info in instance_infos.items():
        obj_stem = Path(geom_obj_file).stem
        bin_path = ScenePath / f"{obj_stem}.bin"
        transform_util.write_transforms(bin_path, copy_info.transforms)

    # Set up the variables needed for codegen
    obj_archives = find_all_archives(element_json)

    base_obj_paths = []
    element_instances_bin_paths = []
    primitive_instances_bin_paths = []
    primitive_instances_handle_indices = []
    curve_records_by_element_instance = []

    for geom_obj_file, instance_info in instance_infos.items():
        obj_stem = Path(geom_obj_file).stem
        bin_path = ScenePath / f"{obj_stem}.bin"

        base_obj_paths.append(geom_obj_file)
        element_instances_bin_paths.append(bin_path)

        primitive_instance_bin_paths = [
            bin_path
            for _, bin_path
            in instance_info.transform_bins
        ]
        primitive_instances_bin_paths.append(primitive_instance_bin_paths)

        primitive_instance_handle_indices = [
            obj_archives.index(obj_path)
            for obj_path, _
            in instance_info.transform_bins
        ]
        primitive_instances_handle_indices.append(primitive_instance_handle_indices)

        curve_records_by_element_instance.append(instance_info.curve_records[:])

    # Codegen
    if output_cpp:
        print("Writing code:")
        code_dict = code.generate_code(
            element_name,
            sbt_manager,
            base_obj_paths,
            element_instances_bin_paths,
            obj_archives,
            primitive_instances_bin_paths,
            primitive_instances_handle_indices,
            curve_records_by_element_instance,
        )
        for filename, code_str in code_dict.items():
            code_path = Path("../src/") / filename
            print(f"  {code_path}")

            f = open(code_path, "w")
            f.write(code_str)

def run():
    sbt_manager = materials.build_sbt_manager(elements)
    if True:
        print(code.generate_sbt_array(sbt_manager))

    # process_element("isBeach", sbt_manager, output_cpp=True)

    temp_elements = [
        "isBayCedarA1",
        "isBeach",
        "isCoastline",
        "isCoral",
        "isDunesA",
        "isDunesB",
        "isGardeniaA",
        "isHibiscus",
        "isHibiscusYoung",
        "isIronwoodA1",
        # # "isIronwoodB",
        "isKava",
        "isLavaRocks",
        "isMountainA",
        "isMountainB",
        "isNaupakaA",
        "isPalmDead",
        "isPalmRig",
        "isPandanusA",
        # "osOcean",
    ]
    for element in temp_elements:
        if element not in skip_list:
            process_element(element, sbt_manager, output_cpp=True)

    # textures.generate_texture_lookup_code()

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
