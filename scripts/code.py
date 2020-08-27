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
    base_objs,
    element_instances_bin_paths,
    obj_archives,
    primitive_instances_bin_paths,
    primitive_instances_handle_indices,
):
    header_filename = _header_filename_from_element_name(element_name)
    source_filename = _source_filename_from_element_name(element_name)
    return {
        header_filename: _generate_header(element_name),
        source_filename: _generate_src(
            element_name,
            base_objs,
            element_instances_bin_paths,
            obj_archives,
            primitive_instances_bin_paths,
            primitive_instances_handle_indices,
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
    base_objs,
    element_instances_bin_paths,
    obj_archives,
    primitive_instances_bin_paths,
    primitive_instances_handle_indices,
):
    class_name = _class_name_from_element_name(element_name)
    header_filename = _header_filename_from_element_name(element_name)

    base_obj_items = "\n".join([
        f"{' ' * 8}moanaRoot + \"/island/{base_obj}\","
        for base_obj
        in base_objs
    ])

    obj_archives_items = "\n".join([
        f"{' ' * 8}moanaRoot + \"/island/{obj_archive}\","
        for obj_archive
        in obj_archives
    ])

    element_instances_bin_path_items = "\n".join([
        f"{' ' * 8}\"{element_instances_bin_path}\","
        for element_instances_bin_path
        in element_instances_bin_paths
    ])

    primitive_instances_bin_paths_items = "\n".join([
        f"{' ' * 8}{{" + ", ".join([
            f"\"{bin_path}\""
            for bin_path
            in primitive_instance_bin_paths
        ]) + "},"
        for primitive_instance_bin_paths
        in primitive_instances_bin_paths
    ])

    primitive_instances_handle_indices_items = "\n".join([
        f"{' ' * 8}{{" + ", ".join([
            str(index)
            for index
            in primitive_instance_handle_indices
        ]) + "},"
        for primitive_instance_handle_indices
        in primitive_instances_handle_indices
    ])

    return f"""\
#include "{header_filename}"

namespace moana {{

{class_name}::{class_name}()
{{
    const std::string moanaRoot = MOANA_ROOT;

    m_elementName = "{element_name}";

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

    }}

}}
"""
