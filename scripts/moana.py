import json
import struct
import os
from pathlib import Path

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

def process(object_name, object_path, archive_paths, output_cpp=False):
    print(f"Processing: {object_name}")

    object_digest = json.load(open(MoanaPath / object_path))
    instanced_copies = [
        copy["transformMatrix"]
        for copy
        in object_digest.get("instancedCopies", {}).values()
    ] + [ object_digest["transformMatrix"] ]

    write_transforms(
        ScenePath / f"{object_name}-root.bin",
        [ corrected_transform(t) for t in instanced_copies ]
    )

    obj_paths = []
    bin_paths = []

    # All the archives for the object
    for archive_path in archive_paths:
        instance_digest = json.load(open(MoanaPath / archive_path))
        archives = instance_digest.keys()

        # Instances grouped by obj inside an archive
        for archive in archives:
            instance_transforms = instance_digest[archive].values()

            archive_stem = Path(archive).stem
            output_filename = ScenePath / f"{object_name}-{archive_stem}.bin"

            obj_paths.append(archive)
            bin_paths.append(output_filename)

            write_transforms(
                output_filename,
                [ corrected_transform(t) for t in instance_transforms ]
            )

    if output_cpp:
        print(f"{' '*4}const std::vector<std::string> objPaths = {{")
        print("\n".join([
            f"{' ' * 8}moanaRoot + \"/island/{obj_path}\","
            for obj_path in obj_paths
        ]))
        print(f"{' '*4}}};")
        print()
        print(f"{' '*4}const std::vector<std::string> binPaths = {{")
        print("\n".join([
            f"{' ' * 8}\"{bin_path}\","
            for bin_path in bin_paths
        ]))
        print(f"{' '*4}}};")


def run():
    process(
        "hibiscus",
        "json/isHibiscus/isHibiscus.json",
        [ "json/isHibiscus/isHibiscus_xgBonsai.json" ],
    )
    process(
        "mountainA",
        "json/isMountainA/isMountainA.json",
        [
            "json/isMountainA/isMountainA_xgBreadFruit.json",
            "json/isMountainA/isMountainA_xgCocoPalms.json",
        ],
    )
    process(
        "mountainB",
        "json/isMountainB/isMountainB.json",
        [
            "json/isMountainB/isMountainB_xgFoliageB.json",
            "json/isMountainB/isMountainB_xgFoliageC.json",
            "json/isMountainB/isMountainB_xgFoliageA.json",
            "json/isMountainB/isMountainB_xgFoliageAd.json",
            "json/isMountainB/isMountainB_xgBreadFruit.json",
            "json/isMountainB/isMountainB_xgCocoPalms.json",
            "json/isMountainB/isMountainB_xgFern.json"
        ],
    )
    process(
        "dunesA",
        "json/isDunesA/isDunesA.json",
        [
            "json/isDunesA/isDunesA_xgPalmDebris.json",
            "json/isDunesA/isDunesA_xgDebris.json",
            "json/isDunesA/isDunesA_xgHibiscusFlower.json",
            "json/isDunesA/isDunesA_xgMuskFern.json",
        ],
    )
    process(
        "ironwoodA1",
        "json/isIronwoodA1/isIronwoodA1.json",
        [
            "json/isIronwoodA1/isIronwoodA1_xgBonsai.json",
        ],
    )
    process(
        "coastline",
        "json/isCoastline/isCoastline.json",
        [
            "json/isCoastline/isCoastline_xgPalmDebris.json",
            "json/isCoastline/isCoastline_xgFibers.json",
        ],
        output_cpp=True
    )

if __name__ == "__main__":
    run()
