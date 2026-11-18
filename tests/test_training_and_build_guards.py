"""Argument and build-path checks without dataset downloads or vendor tools."""
import ast
import os
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_numeric_range_audit_counts_cast_overflow_before_clamping():
    import torch
    tree = ast.parse((ROOT / "scripts/audit_numeric_ranges.py").read_text(encoding="utf-8"))
    definition = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "cast")
    scope = dict(torch=torch, stats={})
    exec(compile(ast.Module(body=[definition], type_ignores=[]), "audit_numeric_ranges.py", "exec"), scope)
    values = torch.tensor([-2048., 2047.9375, -2048.0625, 2048., .09375])
    result = scope["cast"]("input", values)
    assert scope["stats"]["input"]["q12_4_overflow_values"] == 2
    assert scope["stats"]["input"]["values_checked"] == 5
    torch.testing.assert_close(result, torch.tensor([-2048., 2047.9375, -2048., 2047.9375, .0625]))
    with pytest.raises(ValueError, match="Nonfinite"):
        scope["cast"]("bad", torch.tensor([float("nan")]))
    assert "bad" not in scope["stats"]


def gpu_notebook_source():
    from IPython.core.inputtransformer2 import TransformerManager
    notebook = json.loads((ROOT / "training/gpu_benchmarking.ipynb").read_text(encoding="utf-8"))
    return "\n".join(TransformerManager().transform_cell("".join(cell["source"]))
                     for cell in notebook["cells"] if cell["cell_type"] == "code")


def test_gpu_notebook_counts_macs_directly_for_batches():
    import torch
    tree = ast.parse(gpu_notebook_source())
    definition = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and
                      node.name == "count_flops_and_params")
    scope = dict(torch=torch, nn=torch.nn, DEVICE=torch.device("cpu"))
    exec(compile(ast.Module(body=[definition], type_ignores=[]), "gpu_notebook", "exec"), scope)
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 2, 3, padding=1), torch.nn.Flatten(),
                                torch.nn.Linear(128, 4)).eval()
    flops, params, macs = scope["count_flops_and_params"](model, (3, 3, 8, 8))
    assert macs == 11904
    assert flops == 24204
    assert params == 572
    assert all(not module._forward_hooks for module in model.modules())


def test_gpu_notebook_removes_profiling_hooks_after_failure():
    import torch
    tree = ast.parse(gpu_notebook_source())
    definition = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and
                      node.name == "count_flops_and_params")
    scope = dict(torch=torch, nn=torch.nn, DEVICE=torch.device("cpu"))
    exec(compile(ast.Module(body=[definition], type_ignores=[]), "gpu_notebook", "exec"), scope)

    class FailingLayer(torch.nn.Module):
        def forward(self, x):
            raise RuntimeError("deliberate forward failure")

    model = torch.nn.Sequential(torch.nn.Conv2d(3, 2, 3, padding=1), FailingLayer()).train()
    with pytest.raises(RuntimeError, match="deliberate"):
        scope["count_flops_and_params"](model, (1, 3, 8, 8))
    assert model.training
    assert all(not module._forward_hooks for module in model.modules())


def test_gpu_notebook_unknown_gpu_does_not_report_t4_efficiency(capsys):
    source = gpu_notebook_source()
    section = source[source.index("GPU_SPECS ="):source.index("# 6.5 Summary Table")]
    scope = dict(torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
                 gpu_props=SimpleNamespace(name="NVIDIA GeForce RTX 3050 Laptop GPU"),
                 throughput_results=[{"gflops": 100}], total_params=100, total_flops=1000)
    exec(compile(section, "gpu_notebook", "exec"), scope)
    assert scope["gpu_peak_tflops"] is None
    assert scope["compute_efficiency"] is None
    assert scope["bottleneck"] == "Unknown"
    assert "Unavailable" in capsys.readouterr().out


def training_validator():
    source = ast.parse((ROOT / "training/train_replacement.py").read_text(encoding="utf-8"))
    definition = next(node for node in source.body if isinstance(node, ast.FunctionDef) and
                      node.name == "validate_training_args")
    scope = {"Path": Path}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), "train_replacement.py", "exec"), scope)
    return scope["validate_training_args"]


@pytest.mark.parametrize("overrides", [
    {"epochs": 0, "qat_epochs": 0}, {"epochs": -1}, {"batch_size": 0},
    {"threads": 0}, {"epochs": 0, "qat_epochs": 1, "resume": None},
])
def test_invalid_training_configuration_fails_before_dataset_access(overrides):
    args = dict(epochs=20, qat_epochs=5, batch_size=128, threads=4, resume=None)
    args.update(overrides)
    with pytest.raises(ValueError):
        training_validator()(SimpleNamespace(**args))


@pytest.fixture(scope="module")
def tcl_interpreter():
    tkinter = pytest.importorskip("tkinter")
    return tkinter.Tcl()


@pytest.mark.parametrize("mode,csim,csynth,cosim", [
    ("csim", 1, 0, 0), ("csynth", 0, 1, 0), ("cosim", 1, 1, 1),
])
def test_vitis_tcl_resolves_files_from_other_working_directory(tmp_path, mode, csim, csynth, cosim, tcl_interpreter):
    interpreter = tcl_interpreter
    calls = []
    names = ["open_project", "set_top", "add_files", "open_solution", "set_part", "create_clock",
             "config_interface", "config_compile", "config_schedule", "config_dataflow",
             "csim_design", "csynth_design", "export_design", "cosim_design", "exit", "puts"]
    for name in names:
        interpreter.createcommand(name, lambda *args, name=name: calls.append((name, args)))
    previous = Path.cwd()
    try:
        interpreter.call("cd", str(tmp_path))
        interpreter.setvar("argv", mode)
        interpreter.call("source", str(ROOT / "hls/vitis_hls.tcl"))
        assert Path.cwd().resolve() == (ROOT / "hls").resolve()
        assert sum(name == "csim_design" for name, _ in calls) == csim
        assert sum(name == "csynth_design" for name, _ in calls) == csynth
        assert sum(name == "cosim_design" for name, _ in calls) == cosim
        for name, args in calls:
            if name == "add_files":
                assert Path(args[-1]).exists()
    finally:
        interpreter.call("cd", str(previous))
