import json
import struct
import os
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

    has_element_instances = False
    if len(instanced_copies) > 1:
        has_element_instances = True

    obj_paths = []
    bin_paths = []

    archive_paths = []
    for _, instanced_primitive in element_digest["instancedPrimitiveJsonFiles"].items():
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
        print(code.generate_header(element_name))
        print(code.generate_src(element_name, obj_paths, bin_paths, has_element_instances))


def run():
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
        "isMountainA",
        "isMountainB",
    ]
    process("isHibiscusYoung", output_cpp=True)
    # for element in elements:
    #     process(element)

if __name__ == "__main__":
    run()
