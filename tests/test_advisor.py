"""The data advisor recognises each synthetic modality and structure."""

import numpy as np
import pytest

from embedders_backbones import synthetic as S
from embedders_backbones.advisor import analyze_data, suggest, advise, detect_time_axis


@pytest.fixture(scope="module")
def ring():
    return S.make_dataset(n_neurons=50, duration=500, seed=0)


def test_spike_times_ring_population(ring):
    p = analyze_data(ring["spike_times"])
    assert p.modality == "spike_times" and p.kind == "spike_times"
    assert p.has_ring and not p.has_clusters
    r = suggest(p)
    assert r.fit_method == "fit_spikes" and r.embedder_kwargs["bin_size"] == p.bin_size
    assert r.embedder_kwargs["likelihood"] == "gaussian"     # benchmark: no better objective
    assert "pca" in r.compare and "persistence barcode" in r.visuals


def test_counts_matrix_and_transposed_orientation(ring):
    a = analyze_data(ring["counts"])
    b = analyze_data(ring["counts"].T)
    assert a.modality == b.modality == "binned_counts"
    assert a.time_axis == 1 and b.time_axis == 0
    assert a.n_frames == b.n_frames == ring["counts"].shape[1]


def test_metastable_states_are_clusters_not_rings():
    m = S.make_metastable(seed=1)
    p = analyze_data(m["counts"])
    assert p.has_clusters and not p.has_ring and p.k_best == 4
    r = suggest(p)
    assert r.embedder_kwargs["n_states"] == 4 and r.embedder_kwargs["state_model"] == "hmm"
    assert r.embedder_kwargs["likelihood"] == "auto" and r.embedder_kwargs["mask_mode"] == "both"


def test_sheet_has_neither_ring_nor_clusters():
    p = analyze_data(S.make_sheet(seed=0)["counts"])
    assert not p.has_ring and not p.has_clusters


def test_photometry_bleaching_and_dff():
    ph = S.make_photometry(seed=0)
    p = analyze_data(ph["signal"], fs=ph["fs"])
    assert p.modality == "photometry" and p.bleaching and p.drift_dominated
    r = suggest(p)
    assert r.fit_method == "fit_fiber" and r.fit_kwargs["dff"] is True
    assert r.embedder_kwargs["embedding_dim"] <= 8


def test_pose_detected_from_names_and_from_nan_pattern():
    po = S.make_pose(seed=0)
    by_names = analyze_data(po["X"], columns=po["columns"], fs=po["fps"])
    by_nans = analyze_data(po["X"], fs=po["fps"])
    assert by_names.modality == by_nans.modality == "pose"
    assert by_nans.time_axis == 0
    r = suggest(by_nans)
    assert r.embedder_kwargs["state_model"] == "arhmm"


def test_calcium_traces():
    ca = S.make_calcium(seed=0)
    p = analyze_data(ca["traces"], fs=ca["fs"])
    assert p.modality == "calcium" and p.has_ring


def test_behaviour_profiling_and_alignment_warning(ring):
    beh = ring["behavior"]
    p = analyze_data(ring["spike_times"], behavior=beh, behavior_times=ring["times"])
    assert p.behavior["phase"]["task"] == "circular"
    assert p.behavior["state"]["task"] == "classification"
    assert p.behavior["speed"]["task"] == "regression"
    r = suggest(p)
    assert r.decoding["circular"] == ["phase"] and r.decoding["n_perm"] >= 200
    p2 = analyze_data(ring["spike_times"], behavior={"x": np.arange(10.0)})
    assert any("behavior_times" in w for w in suggest(p2).warnings)


def test_report_and_fit_roundtrip(tmp_path):
    m = S.make_metastable(duration=300, seed=2)
    rec = advise(m["counts"])
    txt = rec.report()
    assert "Recommended pipeline" in txt and "Why" in txt
    rec.save(str(tmp_path / "advice"))
    assert (tmp_path / "advice.md").exists() and (tmp_path / "advice.json").exists()
    emb = rec.fit(m["counts"], speed="fast", max_steps=10, state_model="auto",
                  time_budget_s=None, log=None)
    assert emb.Z.shape[0] == m["counts"].shape[1] and emb.recommendation is rec


def test_time_axis_ambiguous_square_matrix_is_low_confidence():
    rng = np.random.default_rng(0)
    axis, conf = detect_time_axis(rng.normal(size=(200, 200)))
    assert conf == "low"
