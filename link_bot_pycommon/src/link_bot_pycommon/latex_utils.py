def make_cell(text, table_format):
    if isinstance(text, list):
        if table_format == 'latex_raw':
            return "\\makecell{" + "\\\\".join(text) + "}"
        else:
            return "\n".join(text)
    else:
        return text