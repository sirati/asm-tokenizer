"""Pair CSV-output BinaryInfo entries with their corresponding mapping
BinaryInfo entries by identifier.

Single source of truth so both the standalone CLI
(`tokenizer.memmap_builder.__main__`) and the dynrunner driver
(`dynrunner.build_memmap.memmap_builder_task`) call the same code.
"""


def match_csv_to_mapping(csv_binaries, mapping_binaries):
    """Match each CSV binary to its corresponding mapping file.

    Returns dict mapping csv BinaryInfo to mapping BinaryInfo.
    """
    matched = {}
    unmatched_csv = []

    for csv_bin in csv_binaries:
        found = False
        for map_bin in mapping_binaries:
            if csv_bin.identifier == map_bin.identifier:
                matched[csv_bin.identifier] = map_bin
                found = True
                break
        if not found:
            unmatched_csv.append(csv_bin)

    return matched, unmatched_csv
