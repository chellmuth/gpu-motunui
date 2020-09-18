import struct

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

def write_transforms(filename, transforms, skip_correction=False):
    print("Writing:", filename)
    print("  Transform count:", len(transforms))
    output_file = open(filename, "wb")

    count_bin = struct.pack("i", len(transforms))
    output_file.write(count_bin)

    for transform in transforms:
        if not skip_correction:
            transform = corrected_transform(transform)

        transform_bin = struct.pack("12f", *transform)
        output_file.write(transform_bin)

if __name__ == "__main__":
    transform = [
        0.9063077870366499,
        0.0,
        -0.4226182617406995,
        0.0,
        -0.2716537822741844,
        0.7660444431189781,
        -0.5825634160695854,
        0.0,
        0.32374437096706465,
        0.6427876096865393,
        0.694272044014884,
        0.0,
        95000.0,
        195000.0,
        200000.0,
        1.0
    ]

    write_transforms("../scene/fixme-transform.bin", [transform])
