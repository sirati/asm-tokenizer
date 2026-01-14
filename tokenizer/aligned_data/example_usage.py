#!/usr/bin/env python3
"""
Example usage of the AlignedDataLoader for loading function data.

OVERVIEW
========
The AlignedDataLoader provides efficient access to assembly function data that has been
processed and aligned across multiple compilation versions (different architectures,
compilers, and optimization levels).

DATA STRUCTURE
==============
The loader works with files created by the export process:
  - {binary}_sections.csv    : Function metadata (architecture, compiler, calls, etc.)
  - {binary}_data.bin        : Binary-packed token/runlength data
  - {binary}_index.bin       : Fast lookup index sorted by function length
  - {binary}_unmatched_*     : Functions not present in all compilation versions

KEY CONCEPTS
============
Matched Functions:
  - Functions that appear in ALL compilation versions
  - Useful for cross-version analysis, compiler comparison, training translation models
  - Each MatchedFunction contains multiple FunctionData objects (one per version)

Unmatched Functions:
  - Functions that appear in only some compilation versions
  - Functions longer than 4096 tokens (too long for standard processing)
  - Single FunctionData objects

MAIN FEATURES
=============
1. Multi-binary support: Load from multiple binaries simultaneously
2. Length filtering: Filter functions by token count (min_length, max_length)
3. Efficient sampling: O(1) lookups using pre-computed edge indices (no searching!)
4. ML-safe design: No memmap caching (prevents memory leaks during training)
5. Three loading modes:
   a) load_matched_functions(n, target_length): N matched functions of similar length
   b) load_unmatched_functions(n): N random unmatched functions
   c) load_random_sections(n): N random mixed sections (unbiased matched/unmatched split)

DESIGN PRINCIPLES
=================
1. NO MEMMAP CACHING: Memmaps are opened fresh in each loading function and closed
   immediately. This prevents the well-known memory leak issues in ML training loops.

2. PRE-COMPUTED EDGE INDICES: During initialization, we build two arrays:
   - edge_indices[L]: First function index where length >= L
   - count_per_length[L]: Number of functions with exactly length L

   Since data is sorted by length, this enables O(1) lookups without searching.

   When sampling N functions at target_length:
   - If not enough at exact length, automatically expand to nearby lengths
   - Uses exponential expansion (±16, ±24, ±36, ...) to find sufficient functions
   - All lookups use pre-computed arrays, no linear search

3. EFFICIENT STORAGE: Only simple numpy arrays are cached, never memmaps. This keeps
   memory usage low while maintaining fast access.

TYPICAL USAGE
=============
    from tokenizer.aligned_data import AlignedDataLoader

    # Initialize
    loader = AlignedDataLoader(
        base_path="/path/to/data",
        binary_names=["minigzipsh", "bzip2"],
        min_length=50,
        max_length=500,
        seed=42
    )

    # Load matched functions for cross-version analysis
    functions = loader.load_matched_functions(n=10, target_length=128)
    for func in functions:
        for version in func.versions:
            print(f"{func.func_name}: {version.metadata['arch']}-{version.metadata['compiler']}")
            print(f"  Tokens: {version.tokens[:10]}...")

    # Load random sections for ML training
    batch = loader.load_random_sections(n=32)  # Mix of matched/unmatched

    # Get dataset statistics
    stats = loader.get_statistics()
    print(f"Total functions: {stats['total_matched_functions']}")

EXAMPLES BELOW
==============
This script demonstrates:
1. Loading matched functions (same function across multiple compilations)
2. Loading unmatched functions (single version functions)
3. Loading random mixed sections
4. Filtering by token length
5. Inspecting function metadata and structure
6. Batch processing for machine learning
"""

from pathlib import Path

from tokenizer.aligned_data import AlignedDataLoader, FunctionData, MatchedFunction


def print_separator(title: str = ""):
    """Print a visual separator."""
    print("\n" + "=" * 80)
    if title:
        print(f"  {title}")
        print("=" * 80)
    print()


def example_basic_usage():
    """Basic usage example."""
    print_separator("BASIC USAGE")

    # Initialize the data loader
    # Point to the directory containing your aligned data files
    base_path = Path("path/to/your/aligned/data")

    # List of binaries to load (e.g., ['minigzipsh', 'bzip2', 'another_binary'])
    binary_names = ["minigzipsh"]

    # Create loader with token length filters
    loader = AlignedDataLoader(
        base_path=base_path,
        binary_names=binary_names,
        min_length=10,  # Only functions with >= 10 tokens
        max_length=1000,  # Only functions with <= 1000 tokens
        seed=42,  # For reproducibility
    )

    # Get statistics
    stats = loader.get_statistics()
    print("Dataset Statistics:")
    print(f"  Total binaries: {stats['total_binaries']}")
    print(f"  Total matched functions: {stats['total_matched_functions']}")
    print(f"  Total unmatched functions: {stats['total_unmatched_functions']}")
    print(f"  Length range: {stats['min_length']} - {stats['max_length']} tokens")

    for binary, binary_stats in stats["binaries"].items():
        print(f"\n  {binary}:")
        print(f"    Matched (total): {binary_stats['total_matched']}")
        print(f"    Matched (in range): {binary_stats['matched_in_range']}")
        print(f"    Unmatched (total): {binary_stats['total_unmatched']}")
        print(f"    Unmatched (in range): {binary_stats['unmatched_in_range']}")


def example_load_matched_functions():
    """Example of loading matched functions."""
    print_separator("LOADING MATCHED FUNCTIONS")

    base_path = Path("path/to/your/aligned/data")
    loader = AlignedDataLoader(
        base_path=base_path, binary_names=["minigzipsh"], min_length=50, max_length=500
    )

    # Load 5 matched functions of similar length
    print("Loading 5 matched functions of similar length...")
    functions = loader.load_matched_functions(n=5)

    for i, func in enumerate(functions, 1):
        print(f"\n--- Function {i}: {func.func_name} ---")
        print(f"  Number of versions: {len(func.versions)}")
        print(f"  Average length: {len(func)} tokens")

        for version in func.versions:
            print(
                f"    - {version.metadata['arch']}-{version.metadata['compiler']}-"
                f"{version.metadata['opt']}: {len(version)} tokens"
            )
            print(f"      Calls: {version.metadata['called'][:3]}...")  # First 3 calls
            print(f"      First 10 tokens: {version.tokens[:10]}")
            print(f"      Instructions: {len(version.insn_runlength)}")
            print(f"      Basic blocks: {len(version.block_runlength)}")


def example_load_specific_length():
    """Example of loading functions of a specific length."""
    print_separator("LOADING FUNCTIONS OF SPECIFIC LENGTH")

    base_path = Path("path/to/your/aligned/data")
    loader = AlignedDataLoader(base_path=base_path, binary_names=["minigzipsh"])

    # Load 3 functions with approximately 128 tokens
    target_length = 128
    print(f"Loading functions with ~{target_length} tokens...")
    functions = loader.load_matched_functions(n=3, target_length=target_length)

    for func in functions:
        actual_lengths = [len(v) for v in func.versions]
        print(
            f"  {func.func_name}: {min(actual_lengths)}-{max(actual_lengths)} tokens "
            f"(avg: {len(func)})"
        )


def example_load_unmatched_functions():
    """Example of loading unmatched functions."""
    print_separator("LOADING UNMATCHED FUNCTIONS")

    base_path = Path("path/to/your/aligned/data")
    loader = AlignedDataLoader(
        base_path=base_path, binary_names=["minigzipsh"], min_length=20, max_length=200
    )

    # Load 10 random unmatched functions
    print("Loading 10 random unmatched functions...")
    functions = loader.load_unmatched_functions(n=10)

    for i, func in enumerate(functions, 1):
        print(f"{i}. {func.func_name}: {len(func)} tokens")
        print(f"   First 15 tokens: {func.tokens[:15]}")
        print(
            f"   Instructions: {len(func.insn_runlength)}, Blocks: {len(func.block_runlength)}"
        )


def example_load_random_sections():
    """Example of loading random mixed sections."""
    print_separator("LOADING RANDOM MIXED SECTIONS")

    base_path = Path("path/to/your/aligned/data")
    loader = AlignedDataLoader(
        base_path=base_path, binary_names=["minigzipsh"], min_length=30, max_length=300
    )

    # Load 20 random sections (mix of matched and unmatched)
    print("Loading 20 random sections (automatically mixed)...")
    sections = loader.load_random_sections(n=20)

    matched_count = sum(1 for s in sections if isinstance(s, MatchedFunction))
    unmatched_count = sum(1 for s in sections if isinstance(s, FunctionData))

    print(f"Loaded {matched_count} matched and {unmatched_count} unmatched functions")
    print("\nSample of loaded sections:")

    for i, section in enumerate(sections[:5], 1):  # Show first 5
        if isinstance(section, MatchedFunction):
            print(f"{i}. [MATCHED] {section.func_name}")
            print(f"   {len(section.versions)} versions, avg {len(section)} tokens")
        else:  # FunctionData
            print(f"{i}. [UNMATCHED] {section.func_name}")
            print(f"   {len(section)} tokens")


def example_multiple_binaries():
    """Example of loading from multiple binaries."""
    print_separator("LOADING FROM MULTIPLE BINARIES")

    base_path = Path("path/to/your/aligned/data")
    loader = AlignedDataLoader(
        base_path=base_path,
        binary_names=["minigzipsh", "bzip2", "gzip"],  # Multiple binaries
        min_length=50,
        max_length=500,
    )

    stats = loader.get_statistics()

    print(f"Loaded {stats['total_binaries']} binaries")
    print(
        f"Total matched functions across all binaries: {stats['total_matched_functions']}"
    )
    print(
        f"Total unmatched functions across all binaries: {stats['total_unmatched_functions']}"
    )

    # Load functions from all binaries
    functions = loader.load_matched_functions(n=10)

    # Count functions per binary
    from collections import Counter

    binary_counts = Counter()

    for func in functions:
        # The first version's metadata won't tell us the binary, but we can track it
        # In practice, you might want to track this differently
        binary_counts["mixed"] += 1

    print("\nLoaded 10 functions from across all binaries")


def example_analyze_function_structure():
    """Example of analyzing function structure in detail."""
    print_separator("ANALYZING FUNCTION STRUCTURE")

    base_path = Path("path/to/your/aligned/data")
    loader = AlignedDataLoader(base_path=base_path, binary_names=["minigzipsh"])

    # Load one function
    functions = loader.load_matched_functions(n=1)
    if not functions:
        print("No functions available")
        return

    func = functions[0]
    print(f"Analyzing function: {func.func_name}")
    print(f"Versions: {len(func.versions)}")

    for version in func.versions:
        print(
            f"\n  Version: {version.metadata['arch']}-{version.metadata['compiler']}-{version.metadata['opt']}"
        )
        print(f"    Total tokens: {len(version.tokens)}")
        print(f"    Instructions: {len(version.insn_runlength)}")
        print(f"    Basic blocks: {len(version.block_runlength)}")

        # Compute tokens per instruction
        avg_tokens_per_insn = len(version.tokens) / len(version.insn_runlength)
        print(f"    Avg tokens/instruction: {avg_tokens_per_insn:.2f}")

        # Compute instructions per block
        avg_insn_per_block = len(version.insn_runlength) / len(version.block_runlength)
        print(f"    Avg instructions/block: {avg_insn_per_block:.2f}")

        # Show instruction boundaries
        insn_boundaries = version.insn_runlength.cumsum()
        print(f"    First 5 instruction boundaries: {insn_boundaries[:5]}")

        # Show block boundaries
        block_boundaries = version.block_runlength.cumsum()
        print(f"    First 5 block boundaries: {block_boundaries[:5]}")

        # Inlining information
        if version.metadata["inlining_map"]:
            print(f"    Detected inlining from other versions:")
            for other_version, inlined_list in version.metadata["inlining_map"].items():
                print(f"      {other_version}: {len(inlined_list)} occurrences")


def example_batch_processing():
    """Example of batch processing for machine learning."""
    print_separator("BATCH PROCESSING FOR ML")

    base_path = Path("path/to/your/aligned/data")
    loader = AlignedDataLoader(
        base_path=base_path,
        binary_names=["minigzipsh"],
        min_length=64,
        max_length=256,
        seed=42,
    )

    # Simulate training loop
    batch_size = 32
    num_batches = 5

    print(f"Simulating {num_batches} batches of size {batch_size}...")

    for batch_idx in range(num_batches):
        # Load random sections for this batch
        batch = loader.load_random_sections(n=batch_size)

        # Process batch
        matched_in_batch = sum(1 for s in batch if isinstance(s, MatchedFunction))
        unmatched_in_batch = batch_size - matched_in_batch

        print(
            f"Batch {batch_idx + 1}: {matched_in_batch} matched, {unmatched_in_batch} unmatched"
        )

        # In a real ML scenario, you would:
        # - Extract tokens from each function
        # - Pad/truncate to fixed length
        # - Create training pairs (for contrastive learning, etc.)
        # - Feed to your model


if __name__ == "__main__":
    print("=" * 80)
    print("  AlignedDataLoader Example Usage")
    print("=" * 80)
    print("\nNOTE: Update the 'base_path' and 'binary_names' in each example")
    print("      to match your actual data location.\n")

    # Run examples (comment out those you don't need)
    try:
        example_basic_usage()
        example_load_matched_functions()
        example_load_specific_length()
        example_load_unmatched_functions()
        example_load_random_sections()
        example_multiple_binaries()
        example_analyze_function_structure()
        example_batch_processing()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print(
            "\nPlease update the base_path in the examples to point to your actual data directory."
        )
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
