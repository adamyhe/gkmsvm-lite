"""Tests for gkm-SVM training."""

import numpy as np
import pytest

from gkmsvm import train_gkmsvm


def _random_seqs_str(n, length, rng, gc_bias=0.5):
    bases = "ACGT"
    seqs = []
    for _ in range(n):
        seq = "".join(rng.choice(list(bases)) for _ in range(length))
        seqs.append(seq)
    return seqs


def _random_onehot(n, length, rng):
    seqs = []
    for _ in range(n):
        idx = rng.integers(0, 4, size=length)
        x = np.zeros((4, length), dtype=np.float32)
        x[idx, np.arange(length)] = 1.0
        seqs.append(x)
    return np.stack(seqs)


class TestTrainGkmsvm:
    def test_basic_training_strings(self):
        rng = np.random.default_rng(42)
        pos = _random_seqs_str(20, 20, rng)
        neg = _random_seqs_str(20, 20, rng)

        model = train_gkmsvm(pos, neg, kernel_type="direct", l=7, k=5, C=1.0)

        assert model.num_support_vectors > 0
        assert model.kernel_type == "gkm_cnt"

    def test_basic_training_arrays(self):
        rng = np.random.default_rng(42)
        pos = _random_onehot(20, 20, rng)
        neg = _random_onehot(20, 20, rng)

        model = train_gkmsvm(pos, neg, kernel_type="estimated", l=7, k=5, d=3, C=1.0)

        assert model.num_support_vectors > 0
        assert model.kernel_type == "gkm_esttrunc"

    def test_model_scores_separate_classes(self):
        rng = np.random.default_rng(123)
        pos = ["AAAAAAAAAAAAAAACCCCC"] * 15 + _random_seqs_str(5, 20, rng)
        neg = ["TTTTTTTTTTTTTTTGGGGG"] * 15 + _random_seqs_str(5, 20, rng)

        model = train_gkmsvm(pos, neg, kernel_type="direct", l=7, k=5, C=10.0)

        from gkmsvm.codec import one_hot_encode

        x_pos = one_hot_encode("AAAAAAAAAAAAAAACCCCC")[None, ...]
        x_neg = one_hot_encode("TTTTTTTTTTTTTTTGGGGG")[None, ...]

        score_pos = model(x_pos).item()
        score_neg = model(x_neg).item()

        assert score_pos > score_neg

    def test_model_callable(self):
        rng = np.random.default_rng(42)
        pos = _random_seqs_str(15, 20, rng)
        neg = _random_seqs_str(15, 20, rng)

        model = train_gkmsvm(pos, neg, kernel_type="direct", l=7, k=5, C=1.0)

        x = _random_onehot(3, 20, rng)
        scores = model(x)
        assert scores.shape == (3, 1)

    def test_kernel_params_stored(self):
        rng = np.random.default_rng(42)
        pos = _random_seqs_str(15, 20, rng)
        neg = _random_seqs_str(15, 20, rng)

        model = train_gkmsvm(pos, neg, kernel_type="estimated", l=7, k=5, d=2, C=1.0)

        params = model.kernel_params
        assert params["L"] == 7
        assert params["k"] == 5
        assert params["d"] == 2

    def test_train_and_save_load(self, tmp_path):
        rng = np.random.default_rng(42)
        pos = _random_seqs_str(15, 20, rng)
        neg = _random_seqs_str(15, 20, rng)

        model = train_gkmsvm(pos, neg, kernel_type="direct", l=7, k=5, C=1.0)

        path = tmp_path / "trained.npz"
        model.save(str(path))

        from gkmsvm import load_model

        loaded = load_model(str(path))

        x = _random_onehot(3, 20, rng)
        np.testing.assert_allclose(loaded(x), model(x), atol=1e-6)

    def test_sv_chunk_size_passed(self):
        rng = np.random.default_rng(42)
        pos = _random_seqs_str(15, 20, rng)
        neg = _random_seqs_str(15, 20, rng)

        model = train_gkmsvm(
            pos,
            neg,
            kernel_type="direct",
            l=7,
            k=5,
            C=1.0,
            sv_chunk_size=5,
        )

        assert model.sv_chunk_size == 5

    def test_weighted_kernel_defaults_M_H(self):
        """M and H default to LS-GKM's 50 and 50."""
        rng = np.random.default_rng(42)
        pos = _random_seqs_str(10, 20, rng)
        neg = _random_seqs_str(10, 20, rng)
        model = train_gkmsvm(pos, neg, kernel_type="weighted", l=7, k=5, C=1.0)
        assert (model.kernel.M, model.kernel.H) == (50, 50.0)

    def test_empty_seqs_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            train_gkmsvm([], ["ACGT" * 5], kernel_type="direct", l=7, k=5)

    def test_device_cpu_explicit(self):
        rng = np.random.default_rng(42)
        pos = _random_seqs_str(15, 20, rng)
        neg = _random_seqs_str(15, 20, rng)

        model = train_gkmsvm(
            pos,
            neg,
            kernel_type="direct",
            l=7,
            k=5,
            C=1.0,
            device="cpu",
        )
        assert model.num_support_vectors > 0
        assert isinstance(model.support_sequences, np.ndarray)

    def test_device_invalid_raises(self):
        rng = np.random.default_rng(42)
        pos = _random_seqs_str(10, 20, rng)
        neg = _random_seqs_str(10, 20, rng)

        with pytest.raises(ValueError, match="Unknown device"):
            train_gkmsvm(
                pos,
                neg,
                kernel_type="direct",
                l=7,
                k=5,
                device="tpu",
            )

    def test_nystrom_solver(self):
        rng = np.random.default_rng(42)
        pos = _random_seqs_str(30, 20, rng)
        neg = _random_seqs_str(30, 20, rng)

        model = train_gkmsvm(
            pos,
            neg,
            kernel_type="direct",
            l=7,
            k=5,
            C=1.0,
            solver="nystrom",
            n_components=20,
            device="cpu",
        )
        assert model.num_support_vectors > 0

        x = _random_onehot(5, 20, rng)
        scores = model(x)
        assert scores.shape == (5, 1)

    def test_smo_solver(self):
        rng = np.random.default_rng(42)
        pos = _random_seqs_str(30, 20, rng)
        neg = _random_seqs_str(30, 20, rng)

        model = train_gkmsvm(
            pos,
            neg,
            kernel_type="direct",
            l=7,
            k=5,
            C=1.0,
            solver="smo",
            device="cpu",
        )
        assert model.num_support_vectors > 0

        x = _random_onehot(5, 20, rng)
        scores = model(x)
        assert scores.shape == (5, 1)

    def test_smo_matches_libsvm(self):
        rng = np.random.default_rng(123)
        pos = _random_seqs_str(25, 20, rng)
        neg = _random_seqs_str(25, 20, rng)

        model_gram = train_gkmsvm(
            pos,
            neg,
            kernel_type="direct",
            l=7,
            k=5,
            C=1.0,
            solver="libsvm",
            device="cpu",
        )
        model_smo = train_gkmsvm(
            pos,
            neg,
            kernel_type="direct",
            l=7,
            k=5,
            C=1.0,
            solver="smo",
            device="cpu",
        )

        x = _random_onehot(10, 20, rng)
        scores_gram = model_gram(x)
        scores_smo = model_smo(x)
        np.testing.assert_allclose(scores_smo, scores_gram, atol=0.1)

    def test_csmo_solver_loaded(self):
        """Verify C SMO solver compiles and loads."""
        from gkmsvm.solver import _get_csmo

        lib = _get_csmo()
        assert lib is not None, "C SMO solver failed to compile/load"

    def test_auto_solver_smo_fallback_warns(self):
        """Auto solver warns when falling back to SMO."""
        import warnings

        rng = np.random.default_rng(42)
        pos = _random_seqs_str(15, 20, rng)
        neg = _random_seqs_str(15, 20, rng)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            train_gkmsvm(
                pos,
                neg,
                kernel_type="direct",
                l=7,
                k=5,
                C=1.0,
                solver="auto",
                device="cpu",
                max_gram_gb=0.0,
            )
        smo_warnings = [x for x in w if "SMO" in str(x.message)]
        assert len(smo_warnings) == 1
        assert "nystrom" in str(smo_warnings[0].message).lower()
        assert "downsampling" in str(smo_warnings[0].message).lower()

    def test_device_auto_works(self):
        rng = np.random.default_rng(42)
        pos = _random_seqs_str(15, 20, rng)
        neg = _random_seqs_str(15, 20, rng)

        model = train_gkmsvm(
            pos,
            neg,
            kernel_type="direct",
            l=7,
            k=5,
            C=1.0,
            device="auto",
        )
        assert model.num_support_vectors > 0

    def test_device_auto_without_gpu(self, monkeypatch):
        """CuPy importing on a machine without a usable GPU (a CPU node with an
        old NVIDIA driver library) keeps "auto" on CPU."""
        import types

        from gkmsvm import backend
        from gkmsvm.train import _move_to_device

        def no_device():
            raise RuntimeError("cudaErrorInsufficientDriver")

        stub = types.SimpleNamespace(
            ndarray=type("ndarray", (), {}),
            cuda=types.SimpleNamespace(
                runtime=types.SimpleNamespace(getDeviceCount=no_device)
            ),
        )
        monkeypatch.setattr(backend, "cp", stub)
        monkeypatch.setattr(backend, "HAS_CUPY", True)
        monkeypatch.setattr(backend, "HAS_MLX", False)
        monkeypatch.setattr("gkmsvm.train.HAS_MLX", False)  # imported by name
        backend.cuda_available.cache_clear()
        try:
            assert not backend.cuda_available()
            X = np.zeros((2, 4, 20), dtype=np.float32)
            assert isinstance(_move_to_device(X, "auto", False), np.ndarray)
            rng = np.random.default_rng(0)
            model = train_gkmsvm(
                _random_seqs_str(10, 20, rng),
                _random_seqs_str(10, 20, rng),
                kernel_type="direct",
                l=7,
                k=5,
                device="cpu",
            )
            model._to_device("auto")
            assert not model._on_gpu
        finally:
            backend.cuda_available.cache_clear()


def test_train_strings_with_n_equal_onehot():
    """DNA strings with N train the same model as their one-hot arrays (N all-zero)."""
    import random

    from gkmsvm.codec import one_hot_encode

    rng = random.Random(4)
    pos = ["".join(rng.choice("ACGT") for _ in range(40)) for _ in range(30)]
    neg = ["".join(rng.choice("ACGT") for _ in range(40)) for _ in range(30)]
    pos[::3] = [s[:12] + "NNNNN" + s[17:] for s in pos[::3]]
    kw = dict(kernel_type="estimated", l=6, k=4, d=2, device="cpu")
    a = train_gkmsvm(pos, neg, **kw)
    enc = lambda seqs: np.stack([one_hot_encode(s, allow_n=True) for s in seqs])  # noqa: E731
    b = train_gkmsvm(enc(pos), enc(neg), **kw)
    np.testing.assert_allclose(a.coefficients, b.coefficients, rtol=1e-10, atol=1e-12)
    assert a.bias == pytest.approx(b.bias)


class TestCgroupMemory:
    """_available_memory caps MemAvailable at the cgroup limit (Slurm --mem)."""

    @staticmethod
    def _write(root, rel, files):
        d = root / rel
        d.mkdir(parents=True)
        for name, text in files.items():
            (d / name).write_text(text)

    def test_v1(self, tmp_path):
        from gkmsvm.train import _cgroup_available_memory

        proc = tmp_path / "cgroup"
        proc.write_text("11:cpuset:/slurm/job1\n5:memory:/slurm/job1\n3:cpu,cpuacct:/slurm/job1\n")
        self._write(tmp_path / "sys", "memory/slurm/job1", {
            "memory.limit_in_bytes": str(4 << 30),
            "memory.usage_in_bytes": str(3 << 30),
            "memory.stat": f"cache 5\ntotal_inactive_file {1 << 30}\n",
        })
        assert _cgroup_available_memory(str(proc), str(tmp_path / "sys")) == 2 << 30

    def test_limit_on_ancestor(self, tmp_path):
        """Slurm's layout: the job's cgroup has the limit, the task's doesn't."""
        from gkmsvm.train import _cgroup_available_memory

        proc = tmp_path / "cgroup"
        proc.write_text("5:memory:/slurm/uid_1/job_2/step_batch/task_0\n")
        unlimited = {"memory.limit_in_bytes": "9223372036854771712",
                     "memory.usage_in_bytes": str(1 << 30), "memory.stat": "total_inactive_file 0\n"}
        sys_root = tmp_path / "sys"
        self._write(sys_root, "memory/slurm/uid_1/job_2/step_batch/task_0", unlimited)
        for name, text in {"memory.limit_in_bytes": str(8 << 30),
                           "memory.usage_in_bytes": str(3 << 30),
                           "memory.stat": "total_inactive_file 0\n"}.items():
            (sys_root / "memory/slurm/uid_1/job_2" / name).write_text(text)
        for name, text in unlimited.items():
            (sys_root / "memory" / name).write_text(text)
        assert _cgroup_available_memory(str(proc), str(sys_root)) == 5 << 30

    def test_v2(self, tmp_path):
        from gkmsvm.train import _cgroup_available_memory

        proc = tmp_path / "cgroup"
        proc.write_text("0::/system.slice/job\n")
        self._write(tmp_path / "sys", "system.slice/job", {
            "memory.max": str(2 << 30),
            "memory.current": str(1 << 30),
            "memory.stat": "anon 1\ninactive_file 0\n",
        })
        assert _cgroup_available_memory(str(proc), str(tmp_path / "sys")) == 1 << 30

    @pytest.mark.parametrize("version", ["v1", "v2"])
    def test_no_limit(self, tmp_path, version):
        from gkmsvm.train import _cgroup_available_memory

        proc = tmp_path / "cgroup"
        if version == "v1":
            proc.write_text("5:memory:/\n")
            files = {"memory.limit_in_bytes": "9223372036854771712",
                     "memory.usage_in_bytes": "1", "memory.stat": "total_inactive_file 0\n"}
            self._write(tmp_path / "sys", "memory", files)
        else:
            proc.write_text("0::/\n")
            files = {"memory.max": "max", "memory.current": "1", "memory.stat": "inactive_file 0\n"}
            (tmp_path / "sys").mkdir()
            for name, text in files.items():
                (tmp_path / "sys" / name).write_text(text)
        assert _cgroup_available_memory(str(proc), str(tmp_path / "sys")) is None

    def test_missing_files(self, tmp_path):
        from gkmsvm.train import _cgroup_available_memory

        assert _cgroup_available_memory(str(tmp_path / "absent"), str(tmp_path)) is None
        proc = tmp_path / "cgroup"
        proc.write_text("5:memory:/slurm/job1\n")
        assert _cgroup_available_memory(str(proc), str(tmp_path / "sys")) is None

    def test_available_memory_is_capped(self, monkeypatch):
        from pathlib import Path

        from gkmsvm import train

        if not Path("/proc/meminfo").exists():
            pytest.skip("no /proc/meminfo")
        monkeypatch.setattr(train, "_cgroup_available_memory", lambda: 12345)
        assert train._available_memory(np) == 12345
        monkeypatch.setattr(train, "_cgroup_available_memory", lambda: None)
        assert train._available_memory(np) > 12345
