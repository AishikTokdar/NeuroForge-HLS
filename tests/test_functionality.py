"""Independent nonzero numerical and failure-path tests; no FPGA required."""
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "weights"))
from model_reference import ReducedVGG, load_model
from convert_weights import convert, quantize_fixed16
from create_test_files import create_test_files
from verify_quantization import verify_all_weights

PTR = ctypes.POINTER(ctypes.c_float)


def ptr(x):
    assert x.dtype == np.float32 and x.flags.c_contiguous
    return x.ctypes.data_as(PTR)


@pytest.fixture(scope="session")
def native(tmp_path_factory):
    out = tmp_path_factory.mktemp("native") / ("probe.dll" if os.name == "nt" else "probe.so")
    command = ["g++", "-std=c++11", "-O2", "-shared", "-DNO_HLS",
               "-Wno-unknown-pragmas", "-I" + str(ROOT / "hls"),
               str(ROOT / "tests/native_probe.cpp"), str(ROOT / "hls/tiled_conv.cpp"),
               str(ROOT / "hls/utils.cpp"), "-o", str(out)]
    command += ["-static-libgcc", "-static-libstdc++"] if os.name == "nt" else ["-fPIC"]
    subprocess.run(command, check=True, capture_output=True, text=True)
    lib = ctypes.CDLL(str(out))
    lib.probe_conv.argtypes = [PTR] * 4 + [ctypes.c_int] * 7
    lib.probe_bn.argtypes = [PTR] * 6 + [ctypes.c_int] * 2
    lib.probe_pool.argtypes = [PTR] * 2 + [ctypes.c_int] * 3
    lib.probe_fc.argtypes = [PTR] * 4 + [ctypes.c_int] * 2
    lib.probe_infer.argtypes = [ctypes.POINTER(PTR)]
    return lib


@pytest.mark.parametrize("m,i,n,p,tile,depth", [
    (3, 11, 5, 1, 8, 4), (2, 7, 9, 0, 4, 8), (1, 3, 1, 1, 1, 8),
    (256, 4, 8, 1, 4, 8), (3, 9, 11, 1, 8, 8)])
def test_tiled_convolution_matches_torch(native, m,i,n,p,tile,depth):
    rng = np.random.default_rng(17)
    x = rng.normal(0, .2, (1,m,i,i)).astype("f4")
    w = rng.normal(0, .2, (n,m,3,3)).astype("f4")
    b = rng.normal(0, .2, n).astype("f4")
    o = i + 2*p - 2
    storage = np.full(n*o*o + 32, 12345, dtype="f4")
    y = storage[16:-16].reshape(1,n,o,o)
    native.probe_conv(ptr(x),ptr(w),ptr(b),ptr(y),m,i,n,o,p,tile,depth)
    expected = F.conv2d(torch.from_numpy(x),torch.from_numpy(w),torch.from_numpy(b),padding=p).numpy()
    np.testing.assert_allclose(y,expected,atol=2e-5,rtol=2e-5)
    assert np.all(storage[:16] == 12345) and np.all(storage[-16:] == 12345)


@pytest.mark.parametrize("variance", [0.0, 1.0, 0.0001])
def test_batchnorm_relu(native, variance):
    rng = np.random.default_rng(11)
    x = rng.normal(0,.1,(1,3,5,5)).astype("f4")
    g = np.array([1., .5, -.5], dtype="f4")
    b = np.array([.1, -.2, .3],dtype="f4")
    mean = np.array([.02,.03,-.01],dtype="f4")
    var = np.full(3,variance,dtype="f4")
    y = np.empty_like(x)
    native.probe_bn(*map(ptr,[x,g,b,mean,var,y]),3,5)
    expected = F.relu(F.batch_norm(torch.from_numpy(x),torch.from_numpy(mean),
                      torch.from_numpy(var),torch.from_numpy(g),torch.from_numpy(b),eps=1e-5)).numpy()
    np.testing.assert_allclose(y,expected,rtol=2e-5,atol=2e-5)
    assert np.isfinite(y).all()


@pytest.mark.parametrize("width", [6, 7])
def test_pool_and_fully_connected(native, width):
    rng=np.random.default_rng(21)
    x=rng.normal(size=(1,3,width,width)).astype("f4")
    y=np.empty((1,3,3,3),dtype="f4")
    native.probe_pool(ptr(x),ptr(y),3,width,3)
    np.testing.assert_array_equal(y,F.max_pool2d(torch.from_numpy(x),2).numpy())
    w=rng.normal(size=(7,27)).astype("f4")
    b=rng.normal(size=7).astype("f4")
    out=np.empty(7,dtype="f4")
    native.probe_fc(*map(ptr,[y,w,b,out]),27,7)
    np.testing.assert_allclose(out,F.linear(torch.from_numpy(y.ravel()),torch.from_numpy(w),
                               torch.from_numpy(b)).numpy(),atol=1e-5,rtol=1e-5)


def synthetic_model():
    torch.manual_seed(9)
    model=ReducedVGG().eval()
    with torch.no_grad():
        for name,t in model.named_parameters():
            if t.ndim > 1:
                t.copy_(torch.round(torch.randn_like(t)*.04*16)/16)
            elif name.endswith("weight"):
                t.fill_(.5)
            else:
                t.copy_(torch.round(torch.randn_like(t)*.1*16)/16)
    return model


def parameter_arrays(model):
    state=model.state_dict()
    names=[]
    for block in range(4):
        for conv,bn in [(0,1),(3,4)]:
            names.extend([f"features.{block}.block.{conv}.weight",f"features.{block}.block.{conv}.bias"])
            names.extend(f"features.{block}.block.{bn}.{suffix}" for suffix in
                         ["weight","bias","running_mean","running_var"])
    names.extend(["classifier.1.weight","classifier.1.bias","classifier.4.weight","classifier.4.bias"])
    return [state[name].numpy().copy() for name in names]


@pytest.mark.parametrize("seed", [1,42,85])
def test_full_inference_nonzero_torch_reference(native,seed):
    model=synthetic_model()
    x=np.random.default_rng(seed).normal(size=(1,3,32,32)).astype("f4")
    out=np.empty(10,dtype="f4")
    arrays=[x]+parameter_arrays(model)+[out]
    native.probe_infer((PTR*len(arrays))(*map(ptr,arrays)))
    with torch.no_grad():
        expected=model(torch.from_numpy(x)).numpy().ravel()
    assert np.any(expected != 0)
    np.testing.assert_allclose(out,expected,atol=5e-5,rtol=5e-5)


def test_parallel_cpp_inference_has_independent_buffers(native):
    from concurrent.futures import ThreadPoolExecutor
    model = synthetic_model()
    weights = parameter_arrays(model)
    inputs = np.random.default_rng(30).normal(size=(8, 3, 32, 32)).astype("f4")

    def infer(i):
        out = np.empty(10, dtype="f4")
        arrays = [inputs[i]] + weights + [out]
        native.probe_infer((PTR * len(arrays))(*map(ptr, arrays)))
        return out

    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = np.stack(list(pool.map(infer, range(len(inputs)))))
    with torch.no_grad():
        expected = model(torch.from_numpy(inputs)).numpy()
    np.testing.assert_allclose(actual, expected, atol=5e-5, rtol=5e-5)


def export_fp32(model, output):
    output.mkdir()
    for name,t in model.state_dict().items():
        if not name.endswith("num_batches_tracked"):
            t.numpy().astype("<f4").tofile(output/(name.replace(".","_")+".bin"))


def test_conversion_reference_and_real_sqnr(tmp_path):
    source=tmp_path/"source"
    output=tmp_path/"params"
    export_fp32(synthetic_model(),source)
    convert(source,output)
    records=verify_all_weights(source,output)
    assert len(records)==52 and all(mse==0 for _,mse,_ in records)
    create_test_files(output,output,"fixed16")
    golden=np.fromfile(output/"output_golden_fp32.bin",dtype="<f4")
    assert golden.size==10 and np.any(golden)
    path=output/"classifier_4_bias.bin"
    path.write_bytes(b"\x00")
    with pytest.raises(ValueError,match="size"):
        verify_all_weights(source,output)


def test_reject_zeroed_export(tmp_path):
    source = tmp_path / "source"
    export_fp32(synthetic_model(), source)
    np.zeros((32, 3, 3, 3), dtype="<f4").tofile(source / "features_0_block_0_weight.bin")
    with pytest.raises(ValueError,match="All-zero"):
        load_model(source)


def test_checkpoint_export_roundtrip(tmp_path):
    from export_checkpoint import export_checkpoint
    model = synthetic_model()
    checkpoint = tmp_path / "model.pth"
    torch.save({"model_state": model.state_dict()}, checkpoint)
    manifest = export_checkpoint(checkpoint, tmp_path / "weights")
    assert len(manifest["files"]["params_int32"]) == 60
    q16 = load_model(tmp_path / "weights/params_int32", "q16")
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, q16.state_dict()[name])


def test_bundled_learned_exports_and_golden(native):
    import hashlib
    manifest_path = ROOT / "weights/export_manifest.json"
    assert manifest_path.is_file(), "Bundled trained export manifest is missing"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for folder, records in manifest["files"].items():
        for filename, expected_hash in records.items():
            path = ROOT / "weights" / folder / filename
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash
    fp32 = load_model(ROOT / "weights/params_fp32")
    q16 = load_model(ROOT / "weights/params_int32", "q16")
    for name, tensor in fp32.state_dict().items():
        if not name.endswith("num_batches_tracked"):
            torch.testing.assert_close(q16.state_dict()[name], tensor, atol=1 / 65536, rtol=0)
    converted = load_model(ROOT / "hls/params", "fixed16")
    x = np.fromfile(ROOT / "hls/params/input_fp32.bin", dtype="<f4").reshape(1, 3, 32, 32)
    expected = np.fromfile(ROOT / "hls/params/output_golden_fp32.bin", dtype="<f4")
    assert np.isfinite(expected).all() and np.any(expected != 0)
    actual = np.empty(10, dtype="f4")
    arrays = [x] + parameter_arrays(converted) + [actual]
    native.probe_infer((PTR * len(arrays))(*map(ptr, arrays)))
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)


def test_quantizer_endpoints():
    x=np.array([-3000,-2048,-.09375,.09375,2047.9375,3000])
    np.testing.assert_array_equal(quantize_fixed16(x),[-32768,-32768,-2,2,32767,32767])
    with pytest.raises(ValueError):
        quantize_fixed16(np.array([np.nan]))


def test_actual_yosys_parser(tmp_path):
    spec=importlib.util.spec_from_file_location("parse_yosys",ROOT/"hls/parse_yosys_report.py")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report=tmp_path/"stat.json"
    report.write_text(json.dumps({"design":{"num_cells_by_type":{
        "LUT6":12,"RAMB36E1":3,"RAMB18E1":2,"DSP48E1":4,"FDRE":25}}}))
    assert module.parse_report(report)=={"LUT":12,"BRAM18 equivalent":8,"DSP":4,"FF":25}
    with pytest.raises(FileNotFoundError):
        module.parse_report(tmp_path/"missing")


@pytest.fixture(scope="session")
def testbench(tmp_path_factory):
    out = tmp_path_factory.mktemp("testbench") / ("sim.exe" if os.name == "nt" else "sim")
    subprocess.run(["g++", "-std=c++11", "-O2", "-DNO_HLS",
                    *[str(ROOT / "hls" / name) for name in
                      ["tb_conv.cpp", "tiled_conv.cpp", "utils.cpp"]], "-o", str(out)],
                   check=True, capture_output=True, text=True)
    return out


@pytest.mark.parametrize("case", ["valid", "wrong_logits", "nan_logits", "nan_input", "negative_variance", "truncated", "missing", "zero_weights", "zero_middle_weights"])
def test_testbench_reports_real_status(testbench, tmp_path, case):
    source = tmp_path / "source"
    output = tmp_path / "params"
    export_fp32(synthetic_model(), source)
    convert(source, output)
    create_test_files(output, output, "fixed16")
    golden = output / "output_golden_fp32.bin"
    if case == "wrong_logits":
        np.full(10, 100, dtype="<f4").tofile(golden)
    elif case == "nan_logits":
        np.full(10, np.nan, dtype="<f4").tofile(golden)
    elif case == "truncated":
        golden.write_bytes(b"\x00")
    elif case == "missing":
        golden.unlink()
    elif case == "zero_weights":
        np.zeros((32, 3, 3, 3), dtype="<i2").tofile(output / "features_0_block_0_weight.bin")
    elif case == "zero_middle_weights":
        np.zeros((64, 64, 3, 3), dtype="<i2").tofile(output / "features_1_block_3_weight.bin")
    elif case == "nan_input":
        np.full(3072, np.nan, dtype="<f4").tofile(output / "input_fp32.bin")
    elif case == "negative_variance":
        np.full(32, -1, dtype="<i2").tofile(output / "features_0_block_1_running_var.bin")
    result = subprocess.run([str(testbench)], cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert (result.returncode == 0) == (case == "valid"), result.stdout + result.stderr
    assert ("TEST PASSED" in result.stdout) == (case == "valid")


@pytest.mark.parametrize("case", ["negative_variance", "erased_q16_weights"])
def test_invalid_checkpoint_does_not_overwrite_existing_export(tmp_path, case):
    from export_checkpoint import export_checkpoint
    model = synthetic_model()
    with torch.no_grad():
        if case == "negative_variance":
            model.features[0].block[1].running_var.fill_(-1)
        else:
            model.features[0].block[0].weight.fill_(1e-8)
    checkpoint = tmp_path / "invalid.pth"
    torch.save({"model_state": model.state_dict()}, checkpoint)
    output = tmp_path / "exports"
    existing = output / "params_fp32/features_0_block_0_weight.bin"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"original export")
    with pytest.raises(ValueError):
        export_checkpoint(checkpoint, output)
    assert existing.read_bytes() == b"original export"
    assert not (output / "params_int32").exists()


def test_notebook_syntax():
    from IPython.core.inputtransformer2 import TransformerManager
    transformer = TransformerManager()
    for path in (ROOT / "training").glob("*.ipynb"):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for i, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile(transformer.transform_cell("".join(cell["source"])), f"{path.name}:{i}", "exec")


@pytest.mark.parametrize("filename", ["vgg_training.ipynb", "gpu_benchmarking.ipynb"])
def test_notebook_model_training_and_quantization_smoke(filename):
    import ast
    from copy import deepcopy
    from IPython.core.inputtransformer2 import TransformerManager

    notebook = json.loads((ROOT / "training" / filename).read_text(encoding="utf-8"))
    scope = {"torch": torch, "nn": torch.nn, "np": np, "deepcopy": deepcopy,
             "DEVICE": torch.device("cpu")}
    wanted = {"VGGBlock", "ReducedVGG", "run_epoch", "train_one_epoch", "evaluate",
              "_qparams_symmetric_per_tensor", "quantize_dequantize_tensor", "quantize_model_weights"}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse(TransformerManager().transform_cell("".join(cell["source"])))
        definitions = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                       and node.name in wanted]
        exec(compile(ast.Module(body=definitions, type_ignores=[]), filename, "exec"), scope)
    model = scope["ReducedVGG"]().eval()
    reference = synthetic_model()
    model.load_state_dict(reference.state_dict())
    images = torch.randn(2, 3, 32, 32)
    labels = torch.tensor([1, 8])
    with torch.no_grad():
        torch.testing.assert_close(model(images), reference(images))
    scope.update(model=model, optimizer=torch.optim.SGD(model.parameters(), lr=.001),
                 criterion=torch.nn.CrossEntropyLoss())
    if "run_epoch" in scope:
        loss, accuracy = scope["run_epoch"]([(images, labels)], model, train=True)
    else:
        loss, accuracy = scope["train_one_epoch"](model, [(images, labels)], scope["criterion"],
                                                scope["optimizer"], scope["DEVICE"])
    assert np.isfinite(loss) and 0 <= accuracy <= 1
    for bits in [8, 16, 32]:
        quantized = scope["quantize_model_weights"](model.eval(), bits)
        with torch.no_grad():
            assert torch.isfinite(quantized(images)).all()


def test_site_local_links():
    from html.parser import HTMLParser
    from urllib.parse import urlsplit, unquote

    class Links(HTMLParser):
        def handle_starttag(self, tag, attrs):
            for key, value in attrs:
                if key in {"src", "href"} and value:
                    target = urlsplit(value)
                    if not target.scheme and not target.netloc and target.path:
                        assert (path.parent / unquote(target.path)).is_file(), (path, value)

    for path in [ROOT / "index.html", *(ROOT / "visualizations").glob("*.html")]:
        Links().feed(path.read_text(encoding="utf-8"))
