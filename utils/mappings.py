from utils.data_utils import FilenameDatasetParser, FolderDatasetParser


def custom_module_mapping(module: str) -> object:

    modules = {
        "folder_parser": FolderDatasetParser,
        "filename_parser": FilenameDatasetParser
    }

    return modules[module]