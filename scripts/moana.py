import json
import struct
import os
from pathlib import Path

MoanaPath = Path(os.environ["MOANA_ROOT"]) / "island"
ScenePath = Path("../scene")

def corrected_transform(transform):
    return [
        transform[0],
        transform[4],
        transform[8],
        transform[12],
        transform[1],
        transform[5],
        transform[9],
        transform[13],
        transform[2],
        transform[6],
        transform[10],
        transform[14],
    ]

def write_transforms(filename, transforms):
    output_file = open(filename, "wb")

    count_bin = struct.pack("i", len(transforms))
    output_file.write(count_bin)

    for transform in transforms:
        transform_bin = struct.pack("12f", *transform)
        output_file.write(transform_bin)

def run():
    instance_digest = json.load(open(MoanaPath / "json/isHibiscus/isHibiscus_xgBonsai.json"))
    archives = instance_digest.keys()

    for archive in archives:
        instance_transforms = instance_digest[archive].values()

        archive_stem = Path(archive).stem
        output_filename = ScenePath / f"hibiscus-{archive_stem}.bin"

        write_transforms(
            output_filename,
            [ corrected_transform(t) for t in instance_transforms ]
        )

    object_digest = json.load(open(MoanaPath / "json/isHibiscus/isHibiscus.json"))
    instanced_copies = [
        copy["transformMatrix"]
        for copy
        in object_digest["instancedCopies"].values()
    ] + [ object_digest["transformMatrix"] ]

    write_transforms(
        ScenePath / "hibiscus-root.bin",
        [ corrected_transform(t) for t in instanced_copies ]
    )

if __name__ == "__main__":
    run()