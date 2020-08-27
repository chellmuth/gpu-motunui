import json
import struct
import os
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
    output_file = open(filename, "wb")

    count_bin = struct.pack("i", len(transforms))
    output_file.write(count_bin)

    for transform in transforms:
        transform_bin = struct.pack("12f", *transform)
        output_file.write(transform_bin)

def process(element_name, output_cpp=False):
    print(f"Processing: {element_name}")

    element_path = f"json/{element_name}/{element_name}.json"

    element_digest = json.load(open(MoanaPath / element_path))
    instanced_copies = [
        copy["transformMatrix"]
        for copy
        in element_digest.get("instancedCopies", {}).values()
    ] + [ element_digest["transformMatrix"] ]

    write_transforms(
        ScenePath / f"{element_name}-root.bin",
        [ corrected_transform(t) for t in instanced_copies ]
    )

    obj_paths = []
    bin_paths = []

    archive_paths = []
    instanced_primitives = element_digest.get("instancedPrimitiveJsonFiles", {})
    for _, instanced_primitive in instanced_primitives.items():
        if instanced_primitive["type"] == "archive":
            archive_paths.append(instanced_primitive["jsonFile"])

    # All the archive instanced primitives for the element
    for archive_path in archive_paths:
        instance_digest = json.load(open(MoanaPath / archive_path))
        archives = instance_digest.keys()

        # Instances grouped by obj inside an archive
        for archive in archives:
            instance_transforms = instance_digest[archive].values()

            archive_stem = Path(archive).stem
            output_filename = ScenePath / f"{element_name}-{archive_stem}.bin"

            obj_paths.append(archive)
            bin_paths.append(output_filename)

            write_transforms(
                output_filename,
                [ corrected_transform(t) for t in instance_transforms ]
            )

    if output_cpp:
        print("Writing code:")
        code_dict = code.generate_code(element_name, obj_paths, bin_paths)
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
    # process("isBeach", output_cpp=True)

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

