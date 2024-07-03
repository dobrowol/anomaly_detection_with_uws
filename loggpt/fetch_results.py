"""Fetch workflow results."""

import logging
import os
import pathlib

import typer
from flytekit.remote import FlyteRemote

app = typer.Typer(help="Fetch loggpt results.")
logger = logging.getLogger(__name__)


@app.command()
def fetch_results(
    domain: str = typer.Option(..., help="Flyte project domain"),
    execution_id: str = typer.Option(..., help="Flyte execution name"),
) -> None:
    # Set output directory
    output_dir = f"./output/{execution_id}"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Connect to Flyte
    flyte_remote = FlyteRemote.from_config(
        default_project="loggpt",
        default_domain=domain,
        config_file_path=os.path.expanduser("~/flyte_sandbox_config"),
    )

    # Fetch execution
    flyte_execution = flyte_remote.fetch_workflow_execution(name=execution_id)
    flyte_remote.sync_workflow_execution(flyte_execution)

    # Check that execution is completed, abort if not completed
    if not flyte_execution.is_complete:
        logger.error(f"Execution {execution_id} is not yet completed")

    # Retrieve workflow name
    workflow = flyte_execution.spec.launch_plan.name

    # Retrieve output
    output = flyte_execution.outputs

    # Fetch output for loggpt.my_workflow.hello_world
    if workflow == "loggpt.my_workflow.hello_world":
        message = output["message"]

        with open(f"{output_dir}/message.txt", "w") as f:
            f.write(message)


if __name__ == "__main__":
    app()
