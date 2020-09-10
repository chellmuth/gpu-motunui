import collections
import enum
import os
from pathlib import Path

MoanaPath = Path(os.environ["MOANA_ROOT"]) / "island"
ScenePath = Path("../scene")

class Root(enum.Enum):
    Moana = 1
    Generated = 2

class RootedPath(collections.namedtuple("RootedPath", ["path", "root"])):
    __slots__ = ()

    @property
    def fs_path(self):
        if self.root == Root.Moana:
            return MoanaPath / self.path
        elif self.root == Root.Generated:
            return ScenePath / self.path
        else:
            raise RuntimeError("Unknown root value")

    @property
    def code_path(self):
        if self.root == Root.Moana:
            return f"moanaRoot + \"/island/{self.path}\""
        elif self.root == Root.Generated:
            return f"\"{ScenePath / self.path}\""
        else:
            raise RuntimeError("Unknown root value")


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
