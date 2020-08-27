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

def generate_code(element_name, obj_paths, bin_paths):
    header_filename = _header_filename_from_element_name(element_name)
    source_filename = _source_filename_from_element_name(element_name)
    return {
        header_filename: _generate_header(element_name),
        source_filename: _generate_src(element_name, obj_paths, bin_paths)
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

def _generate_src(element_name, obj_paths, bin_paths):
    class_name = _class_name_from_element_name(element_name)
    header_filename = _header_filename_from_element_name(element_name)

    obj_path_items = "\n".join([
        f"{' ' * 8}moanaRoot + \"/island/{obj_path}\","
        for obj_path
        in obj_paths
    ])

    bin_path_items = "\n".join([
        f"{' ' * 8}\"{bin_path}\","
        for bin_path
        in bin_paths
    ])

    element_instances_bin_path = f"../scene/{element_name}-root.bin"

    return f"""\
#include "{header_filename}"

namespace moana {{

{class_name}::{class_name}()
{{
    const std::string moanaRoot = MOANA_ROOT;

    m_baseObj = moanaRoot + "/island/obj/{element_name}/{element_name}.obj";

    m_objPaths = {{
{obj_path_items}
    }};

    m_binPaths = {{
{bin_path_items}
    }};

    m_elementInstancesBinPath = "{element_instances_bin_path}";
}}

}}
"""
