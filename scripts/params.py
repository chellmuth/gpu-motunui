import os
from pathlib import Path

MoanaPath = Path(os.environ["MOANA_ROOT"]) / "island"
ScenePath = Path("../scene")

elements = [
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

skip_list = [
    # "isBeach"
]
