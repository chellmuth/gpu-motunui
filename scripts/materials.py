import collections
import json
import pprint

SBTRecord = collections.namedtuple("SBTRecord", [ "name", "base_color"])

class SBTManager:
    def __init__(self):
        self.records_by_element = {}
        self.curve_assignments = {}

    def add_material_digest(self, element, digest_path):
        records = []

        material_json = json.load(open(digest_path))
        for name, properties in material_json.items():
            if name == "hidden":
                base_color = [ 0., 0., 1. ]
            else:
                base_color = properties["baseColor"][:3]

            record = SBTRecord(name, base_color)
            records.append(record)

            for assignee in properties["assignment"]:
                self.curve_assignments[(element, assignee)] = name

        self.records_by_element[element] = records

    def get_base_colors(self):
        result = [ (0., 0., 0.) ] # unmapped default material
        for element_name in sorted(self.records_by_element.keys()):
            element_records = sorted(
                self.records_by_element[element_name],
                key=lambda r: r.name
            )
            print([e.name for e in element_records])
            for record in element_records:
                result.append(record.base_color)

        return result

    def get_names(self, search_element):
        records = self.records_by_element[search_element]
        return sorted(
            record.name
            for record in records
        )

    def get_sbt_offset(self, search_element):
        offset = 1 # account for default material
        for element_name in sorted(self.records_by_element.keys()):
            if element_name == search_element:
                return offset

            offset += len(self.records_by_element[element_name])

        raise ValueError("Invalid search element")

    def get_mtl_index_for_curve(self, element_name, curve_name):
        try:
            material_name = self.curve_assignments[(element_name, curve_name)]
            return self.get_names(element_name).index(material_name)
        except KeyError:
            # fixme
            assert element_name == "isCoastline"
            assert curve_name == "xgGrass"
            return 0
