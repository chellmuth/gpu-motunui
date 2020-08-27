import json
import struct

def reflect(reflected_point, pivot_point):
    return [
        pivot_point[0] + (pivot_point[0] - reflected_point[0]),
        pivot_point[1] + (pivot_point[1] - reflected_point[1]),
        pivot_point[2] + (pivot_point[2] - reflected_point[2]),
    ]

def pin_curve(curve):
    p_first = reflect(curve[1], curve[0])
    p_last = reflect(curve[-2], curve[-1])

    return [ p_first ] + curve + [ p_last ]

def write_curve_bin(curve_info, bin_path):
    print("Writing:", curve_info.json_path)

    curves = json.load(open(curve_info.json_path))
    print("  Curve count:", len(curves))

    f = open(bin_path, "wb")

    curve_lengths = set()
    for curve in curves:
        curve_lengths.add(len(curve))

    # Strand count
    f.write(struct.pack("i", len(curves)))

    # Vertices per strand
    assert(len(curve_lengths) == 1)
    f.write(struct.pack("i", len(curves[0]) + 2)) # phantom points to pin the curve

    # Widths:
    #  1. Root
    #  2. Tip
    f.write(struct.pack("2f", curve_info.width_root, curve_info.width_tip))

    # Vertex data
    for curve in curves:
        pinned_curve = pin_curve(curve)
        for control_point in pinned_curve:
            f.write(struct.pack("3f", *control_point))

# def read_curve():
#     f = open("../scene/curve.bin", "rb")

#     length, = struct.unpack("i", f.read(4))
#     print(length)

#     for _ in range(length):
#         for _ in range(6):
#             x, y, z = struct.unpack("3f", f.read(4 * 3))
#             print(x, y, z)

#         break
