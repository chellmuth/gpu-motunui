from params import MoanaPath, Root, RootedPath

def obj_tokens(obj_filename):
    with open(obj_filename, "r") as obj:
        for line in obj:
            tokens = line.strip().split(" ", 1)
            if not tokens: continue

            command = tokens[0]
            if len(tokens) > 1:
                yield command, tokens[1]
            else:
                yield command, None


def split(obj_path):
    out_filename1 = f"{obj_path.stem}-1.obj"
    out_filename2 = f"{obj_path.stem}-2.obj"

    count = 0
    for _ in obj_tokens(obj_path):
        count += 1

    target = count // 2

    in_section1 = True
    section1_lines = []
    section2_lines = []

    vertex_offset = 0
    normal_offset = 0

    for i, (command, rest) in enumerate(obj_tokens(obj_path)):
        if in_section1 \
           and i > target \
           and command == "g" and rest == "default":

            print(f"Splitting at line {i + 1}")
            in_section1 = False

        if in_section1:
            if command == "v":
                vertex_offset += 1
            elif command == "vn":
                normal_offset += 1

            section1_lines.append(f"{command or ''} {rest or ''}\n")
        else:
            if command == "f":
                v0, n0, v1, n1, v2, n2, v3, n3 = [
                    int(num)
                    for num in rest.replace("/", " ").split(" ")
                    if num
                ]

                rest = " ".join([
                    f"{v0 - vertex_offset}//{n0 - normal_offset}",
                    f"{v1 - vertex_offset}//{n1 - normal_offset}",
                    f"{v2 - vertex_offset}//{n2 - normal_offset}",
                    f"{v3 - vertex_offset}//{n3 - normal_offset}",
                ])

            section2_lines.append(f"{command or ''} {rest or ''}\n")

    out_path1 = RootedPath(out_filename1, Root.Generated)
    with open(out_path1.fs_path, "w") as f:
        f.writelines(section1_lines)

    out_path2 = RootedPath(out_filename2, Root.Generated)
    with open(out_path2.fs_path, "w") as f:
        f.writelines(section2_lines)

    return (out_path1, out_path2)

if __name__ == "__main__":
    split(
        MoanaPath / "obj/isIronwoodA1/isIronwoodA1.obj",
        Path("../scene")
    )
