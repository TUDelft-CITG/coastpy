import json
import pathlib

import msgspec

from coastpy.schema.schema_hooks import custom_schema_hook
from coastpy.schema.types import ModelUnion

if __name__ == "__main__":
    # Determine the script's directory
    script_dir = pathlib.Path(__file__).resolve().parent

    # Construct the output path relative to the script's directory
    outpath = script_dir.parent.parent / "specification/schema/schema.json"

    # Ensure the parent directories exist (optional but safe)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    # Generate the schema using msgspec
    schema_dict = msgspec.json.schema(ModelUnion, schema_hook=custom_schema_hook)

    # Write the schema to the specified path as pretty-printed JSON
    with outpath.open("w") as f:
        json.dump(schema_dict, f, indent=4)
