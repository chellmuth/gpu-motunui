import collections

import files
import params
from obj import obj_tokens

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

def check_independent_meshes(obj_filename):
    vertex_offset = 0
    normal_offset = 0

    vertex_count = 0
    normal_count = 0

    for command, rest in obj_tokens(obj_filename):
        if command == "v":
            vertex_count += 1
        elif command == "n":
            normal_count += 1
        elif command == "g":
            if rest.startswith("default"):
                vertex_offset = vertex_count
                normal_offset = normal_count
        elif command == "f":
            v0, n0, v1, n1, v2, n2, v3, n3 = [
                int(num)
                for num in rest.replace("/", " ").split(" ")
                if num
            ]

            valid_indices = [
                v0 - vertex_offset >= 1,
                v1 - vertex_offset >= 1,
                v2 - vertex_offset >= 1,
                n0 - normal_offset >= 1,
                n1 - normal_offset >= 1,
                n2 - normal_offset >= 1,
            ]

            if not all(valid_indices):
                print(obj_filename)

def check_duplicate_mesh(obj_filename, mesh):
    first_mesh_vertices = []
    first_mesh_normals = []
    first_face_indices = []
    second_face_indices = []

    current_vertices = []
    current_normals = []

    assert_next_line_not_usemtl = False

    found_first_mesh = False
    collect_first_face = False
    collect_second_face = False
    for command, rest in obj_tokens(obj_filename):
        if command == "v":
            current_vertices.append(rest)
        elif command == "n":
            current_normals.append(rest)
        elif command == "g":
            collect_first_face = False

            if collect_second_face:
                break

            if rest.startswith("default"):
                current_vertices = []
                current_normals = []

            elif rest.startswith(mesh):
                if found_first_mesh:
                    conditions = [
                        current_vertices == first_mesh_vertices,
                        current_normals == first_mesh_normals,
                    ]
                    if not all(conditions):
                        return False

                    collect_second_face = True
                    assert_next_line_not_usemtl = True
                    continue
                else:
                    first_mesh_vertices = current_vertices
                    first_mesh_normals = current_normals

                    found_first_mesh = True
                    collect_first_face = True
        elif command == "usemtl" and assert_next_line_not_usemtl:
            return False
        elif command == "f":
            assert not (collect_first_face and collect_second_face)

            if collect_first_face:
                indices = [
                    int(num)
                    for num in rest.replace("/", " ").split(" ")
                    if num
                ]
                first_face_indices.extend(indices)
            elif collect_second_face:
                indices = [
                    int(num)
                    for num in rest.replace("/", " ").split(" ")
                    if num
                ]
                second_face_indices.extend(indices)

        assert_next_line_not_usemtl = False

    assert len(first_face_indices) > 0
    assert len(second_face_indices) > 0

    # Finished parsing the second mesh's faces
    indices_differences = [
        i - j
        for i, j
        in zip(second_face_indices, first_face_indices)
    ]
    vertex_difference = indices_differences[0]
    normal_difference = indices_differences[1]

    return all(
        i % 2 == 0 and diff == vertex_difference or diff == normal_difference
        for i, diff in enumerate(indices_differences)
    )

if __name__ == "__main__":
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

    for element_name in params.elements:
        print(f"Checking independent meshes {element_name}:")
        for obj_filename in files.find_obj_files(element_name):
            check_independent_meshes(obj_filename)

    for element_name in params.elements:
        print(f"Checking double meshes {element_name}:")
        for obj_filename in files.find_obj_files(element_name):
            records = find_double_meshes(obj_filename)
            for record in records:
                print(f"  [{record.obj_filename}] {record.mesh}={record.count}")

                print(f"   Checking dupe: ", end="")
                if check_duplicate_mesh(obj_filename, record.mesh):
                    print("OKAY!")
                else:
                    print("BAD!")
