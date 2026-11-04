"""Execute the RepWise MCP lab with this interpreter and save its outputs."""

import json
from pathlib import Path
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Agentic_04Nov_S3_Lab_Build_and_connect_an_MCP_server.ipynb"


def main():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    with tempfile.TemporaryDirectory(prefix="repwise-lab-kernel-") as temporary:
        kernels = Path(temporary) / "kernels"
        spec = kernels / "repwise-lab"
        spec.mkdir(parents=True)
        (spec / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "RepWise lab", "language": "python",
            "env": {"IPYTHONDIR": str(Path(temporary) / "ipython"),
                    "JUPYTER_RUNTIME_DIR": str(Path(temporary) / "runtime")},
        }), encoding="utf-8")
        manager = KernelManager(
            kernel_name="repwise-lab", kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)])
        )
        client = NotebookClient(
            notebook, km=manager, timeout=480,
            resources={"metadata": {"path": str(ROOT)}},
            on_cell_start=lambda cell, cell_index, **kwargs: print(
                f"Cell {cell_index + 1}/{len(notebook.cells)}: {cell.cell_type}", flush=True
            ),
        )
        try:
            client.execute()
        finally:
            # Preserve diagnostics on failure; do not mistake a partial run for success.
            notebook.metadata.kernelspec = {
                "display_name": "Python 3 (RepWise environment)", "name": "python3", "language": "python",
            }
            nbformat.write(notebook, NOTEBOOK)
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
            manager.cleanup_resources()
    errors = [output for cell in notebook.cells if cell.cell_type == "code"
              for output in cell.outputs if output.output_type == "error"]
    if errors:
        raise RuntimeError("Notebook contains failed cells")
    print(f"Executed notebook: {NOTEBOOK}")


if __name__ == "__main__":
    main()
