import sys
import great_expectations as gx
from pathlib import Path

# context = gx.get_context(context_root_dir=str(Path(__file__).parent))
context = gx.get_context()

datasource_name = "energy_demand"
data_asset_name = "energy_demand_data"

# Override base_directory so it works on any machine
# context.sources.add_or_update_pandas_filesystem(
#     name=datasource_name,
#     base_directory=str(Path(__file__).parent.parent / "data")
# )

asset = context.get_datasource(datasource_name).get_asset(data_asset_name)

# Load checkpoint
checkpoint_name = "energy_demand_checkpoint"
checkpoint = context.get_checkpoint(checkpoint_name)

# Run checkpoint
checkpoint_result = checkpoint.run(run_id="energy_demand_run")

# Build data docs
context.build_data_docs()

# Check if the checkpoint passed
if checkpoint_result["success"]:
    print("Validation passed!")
    sys.exit(0)
else:
    print("Validation failed!")
    sys.exit(1)
