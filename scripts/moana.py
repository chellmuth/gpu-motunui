import json
import struct
import os
from pathlib import Path

MoanaPath = Path(os.environ["MOANA_ROOT"]) / "island"
ScenePath = Path("../scene")

def run():
    instance_digest = json.load(open(MoanaPath / "json/isHibiscus/isHibiscus_xgBonsai.json"))
    archives = instance_digest.keys()

    for archive in archives:
        instance_transforms = instance_digest[archive].values()

        archive_stem = Path(archive).stem
        output_file = open(ScenePath / f"hibiscus-{archive_stem}.bin", "wb")

        count_bin = struct.pack("i", len(instance_transforms))
        output_file.write(count_bin)

        for instance_transform in instance_transforms:
            transform_row_major = [
                instance_transform[0],
                instance_transform[4],
                instance_transform[8],
                instance_transform[12],
                instance_transform[1],
                instance_transform[5],
                instance_transform[9],
                instance_transform[13],
                instance_transform[2],
                instance_transform[6],
                instance_transform[10],
                instance_transform[14],
            ]

            transform_bin = struct.pack("12f", *transform_row_major)
            output_file.write(transform_bin)

if __name__ == "__main__":
    run()
