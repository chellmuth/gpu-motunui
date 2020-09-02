import collections

import files
import params

def obj_tokens(obj_filename):
    with open(obj_filename, "r") as obj:
        for line in obj:
            tokens = line.strip().split()
            if not tokens: continue

            command = tokens[0]
            if len(tokens) > 1:
                yield command, tokens[1]
            else:
                yield command, None

DoubleMeshRecord = collections.namedtuple("DoubleMeshRecord", [
    "obj_filename",
    "mesh",
    "count",
])

def find_double_meshes(obj_filename):
    mesh_counts = collections.defaultdict(int)

    with open(obj_filename, "r") as obj:
        for line in obj:
            tokens = line.strip().split()
            if not tokens: continue
            if tokens[0] == "g":
                mesh = tokens[1]
                if mesh == "default": continue

                mesh_counts[mesh] += 1

    records = []
    for mesh, count in mesh_counts.items():
        if count > 1:
            record = DoubleMeshRecord(
                obj_filename.relative_to(params.MoanaPath),
                mesh,
                count
            )
            records.append(record)

    return records

def check_default_mesh_on_all_geometry(obj_filename):
    current_mesh = None

    for command, rest in obj_tokens(obj_filename):
        if command == "g":
            current_mesh = rest
        elif command in [ "v", "vn" ]:
            if current_mesh != "default":
                return False

    return True


if __name__ == "__main__":
    for element_name in params.elements:
        print(f"Checking double meshes {element_name}:")
        for obj_filename in files.find_obj_files(element_name):
            records = find_double_meshes(obj_filename)
            for record in records:
                print(f"  [{record.obj_filename}] {record.mesh}={record.count}")

    for element_name in params.elements:
        valid = True
        print(f"Checking default mesh use {element_name}: ", end="")
        for obj_filename in files.find_obj_files(element_name):
            if not check_default_mesh_on_all_geometry(obj_filename):
                print()
                print(obj_filename)
                valid = False
        if valid:
            print("OK!")
