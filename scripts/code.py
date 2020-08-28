def _class_name_from_element_name(element_name):
    return f"{element_name[2:]}Element"

def _file_stem_from_element_name(element_name):
    return "".join([
        "_" + letter.lower() if letter.isupper() else letter
        for letter in element_name[2:]
    ]).lstrip("_") + "_element"

def _header_filename_from_element_name(element_name):
    stem = _file_stem_from_element_name(element_name)
    return f"scene/{stem}.hpp"

def _source_filename_from_element_name(element_name):
    stem = _file_stem_from_element_name(element_name)
    return f"scene/{stem}.cpp"

def generate_code(
    element_name,
    sbt_manager,
    base_objs,
    element_instances_bin_paths,
    obj_archives,
    primitive_instances_bin_paths,
    primitive_instances_handle_indices,
    curve_records_by_element_instance,
):
    header_filename = _header_filename_from_element_name(element_name)
    source_filename = _source_filename_from_element_name(element_name)
    return {
        header_filename: _generate_header(element_name),
        source_filename: _generate_src(
            element_name,
            sbt_manager,
            base_objs,
            element_instances_bin_paths,
            obj_archives,
            primitive_instances_bin_paths,
            primitive_instances_handle_indices,
            curve_records_by_element_instance,
        )
    }

def _generate_header(element_name):
    class_name = _class_name_from_element_name(element_name)

    return f"""\
#pragma once

#include "scene/element.hpp"

namespace moana {{

class {class_name} : public Element {{
public:
    {class_name}();
}};

}}
"""

def _generate_src(
    element_name,
    sbt_manager,
    base_objs,
    element_instances_bin_paths,
    obj_archives,
    primitive_instances_bin_paths,
    primitive_instances_handle_indices,
    curve_records_by_element_instance,
):
    class_name = _class_name_from_element_name(element_name)
    header_filename = _header_filename_from_element_name(element_name)

    base_obj_items = "\n".join([
        f"{' ' * 8}moanaRoot + \"/island/{base_obj}\","
        for base_obj
        in base_objs
    ])

    sbt_offset = sbt_manager.get_sbt_offset(element_name)

    mtl_lookup_items = "\n".join(
        f"{' ' * 8}\"{material_name}\","
        for material_name
        in sbt_manager.get_names(element_name)
    )

    obj_archives_items = "\n".join(
        f"{' ' * 8}moanaRoot + \"/island/{obj_archive}\","
        for obj_archive
        in obj_archives
    )

    element_instances_bin_path_items = "\n".join(
        f"{' ' * 8}\"{element_instances_bin_path}\","
        for element_instances_bin_path
        in element_instances_bin_paths
    )

    primitive_instances_bin_paths_items = "\n".join(
        f"{' ' * 8}{{" + ", ".join(
            f"\"{bin_path}\""
            for bin_path
            in primitive_instance_bin_paths
        ) + "},"
        for primitive_instance_bin_paths
        in primitive_instances_bin_paths
    )

    primitive_instances_handle_indices_items = "\n".join(
        f"{' ' * 8}{{" + ", ".join(
            str(index)
            for index
            in primitive_instance_handle_indices
        ) + "},"
        for primitive_instance_handle_indices
        in primitive_instances_handle_indices
    )

    curve_bin_paths_items = "\n".join(
        f"{' ' * 8}{{" + ", ".join(
            f"\"{record.bin_path}\""
            for record
            in curve_records
        ) + "},"
        for curve_records
        in curve_records_by_element_instance
    )

    curve_mtl_indices_items = "\n".join(
        f"{' ' * 8}{{" + ", ".join(
            f"{sbt_manager.get_mtl_index_for_curve(element_name, record.assignment_name)}"
            for record
            in curve_records
        ) + "},"
        for curve_records
        in curve_records_by_element_instance
    )

    return f"""\
#include "{header_filename}"

namespace moana {{

{class_name}::{class_name}()
{{
    const std::string moanaRoot = MOANA_ROOT;

    m_elementName = "{element_name}";

    m_sbtOffset = {sbt_offset};

    m_mtlLookup = {{
{mtl_lookup_items}
    }};

    m_baseObjs = {{
{base_obj_items}
    }};

    m_objArchivePaths = {{
{obj_archives_items}
    }};

    m_elementInstancesBinPaths = {{
{element_instances_bin_path_items}
    }};

    m_primitiveInstancesBinPaths = {{
{primitive_instances_bin_paths_items}
    }};

    m_primitiveInstancesHandleIndices = {{
{primitive_instances_handle_indices_items}
    }};

    m_curveBinPathsByElementInstance = {{
{curve_bin_paths_items}
    }};

    m_curveMtlIndicesByElementInstance = {{
{curve_mtl_indices_items}
    }};

    }}

}}
"""

def generate_sbt_array(sbt_manager):
    colors = sbt_manager.get_base_colors()
    color_items = "\n".join(
        f"{' '*4}float3{{ {c[0]}, {c[1]}, {c[2]} }},"
        for c in colors
    )

    return f"""\
std::vector<float3> baseColors = {{
{color_items}
}};
"""
