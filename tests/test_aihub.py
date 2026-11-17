from __future__ import annotations

from types import SimpleNamespace

from edge_ai_bench import aihub


class FakeJob:
    url = "https://workbench.aihub.qualcomm.com/jobs/abc"

    def __init__(self, ok: bool = True) -> None:
        self.ok = ok

    def wait(self):
        return SimpleNamespace(success=self.ok, message="" if self.ok else "boom")

    def get_target_model(self):
        return "target-model"

    def download_profile(self):
        return {
            "execution_summary": {
                "estimated_inference_time": 1500,
                "all_inference_times": [1000, 1500, 2000],
                "estimated_inference_peak_memory": 4 * 1024**2,
                "warm_load_time": 20000,
            }
        }


class FakeHub:
    def __init__(self, compile_ok: bool = True) -> None:
        self.compile_ok = compile_ok
        self.calls: list[tuple] = []

    def Device(self, name):
        return SimpleNamespace(name=name)

    def submit_compile_job(self, **kw):
        self.calls.append(("compile", kw))
        return FakeJob(self.compile_ok)

    def submit_profile_job(self, **kw):
        self.calls.append(("profile", kw))
        return FakeJob()

    def get_devices(self):
        return [
            SimpleNamespace(
                name="Samsung Galaxy S23 (Family)",
                os="13",
                attributes=["chipset:qualcomm-snapdragon-8gen2"],
            ),
            SimpleNamespace(
                name="Google Pixel 7", os="13", attributes=["chipset:google-tensor-g2"]
            ),
        ]


def test_check_ready_explains_setup(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("QAIHUB_CLIENT_INI", str(tmp_path / "missing.ini"))
    monkeypatch.setattr(aihub, "import_hub", lambda: object())
    hub, why = aihub.check_ready()
    assert hub is None and "qai-hub configure --api_token" in why


def test_profile_maps_results(tiny_models) -> None:
    root, spec = tiny_models
    hub = FakeHub()
    recs = aihub.profile(hub, [spec], root, ["fp32"], "Samsung Galaxy S23 (Family)", "qnn_dlc")
    assert recs[0]["status"] == "ok"
    assert recs[0]["latency"]["p50_ms"] == 1.5
    assert recs[0]["memory"]["peak_rss_mb"] == 4.0
    kind, kw = hub.calls[0]
    assert kind == "compile" and kw["options"] == "--target_runtime qnn_dlc"
    assert kw["input_specs"] == {"x": ((1, 8), "float32")}


def test_profile_records_compile_failure(tiny_models) -> None:
    root, spec = tiny_models
    recs = aihub.profile(FakeHub(compile_ok=False), [spec], root, ["int8"], "dev", "onnx")
    assert recs[0]["status"] == "error" and "compile failed" in recs[0]["error"]


def test_list_devices_filters_by_chipset() -> None:
    devs = aihub.list_devices(FakeHub(), "8gen2")
    assert [d["name"] for d in devs] == ["Samsung Galaxy S23 (Family)"]
