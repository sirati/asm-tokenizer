from pathlib import Path

VERIFICATION: bool = False
SCRIPT_FOLDER: Path = Path(__file__).parent.resolve()


degenerate_prefixes = {
    0xF2: ["repne", "repnz"],
    0xF3: ["repe", "repz", "rep"],  # ordering important due to string comparisons
}
