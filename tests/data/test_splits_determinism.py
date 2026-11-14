"""Split hygiene and determinism under a seed."""

import numpy as np
import pandas as pd

from prahari.db import get_engine
from prahari.pipeline.splits import load_splits
from prahari.sim.faults import RAMP_DOWN
from prahari.sim.generate import build_dataset


def test_splits_disjoint_chronological_and_no_straddle(small_db):
    s, _ = small_db
    sp = load_splits(s)
    inc = pd.read_sql(
        "SELECT incident_id, start_ts, end_ts FROM incidents",
        get_engine(s),
        parse_dates=["start_ts", "end_ts"],
    ).set_index("incident_id")
    ids = [set(sp[k]) for k in ("train", "val", "test")]
    assert not (ids[0] & ids[1] or ids[1] & ids[2] or ids[0] & ids[2])
    assert set().union(*ids) == set(inc.index)
    tail = pd.Timedelta(minutes=RAMP_DOWN)
    for a, b, bound in (("train", "val", "train_end"), ("val", "test", "val_end")):
        cut = pd.Timestamp(sp["boundaries"][bound])
        assert inc.loc[sp[a], "start_ts"].max() < cut <= inc.loc[sp[b], "start_ts"].min()
        # no earlier-split incident (incl. KPI recovery tail) is still active at the boundary
        assert (inc.loc[sp[a], "end_ts"] + tail <= cut).all(), f"{a} incident straddles {bound}"


def test_true_alarms_stay_on_their_incident_side(small_db):
    s, _ = small_db
    sp = load_splits(s)
    eng = get_engine(s)
    al = pd.read_sql(
        "SELECT ts, label_incident_id FROM alarms WHERE label_is_true = 1", eng, parse_dates=["ts"]
    )
    side = {i: k for k in ("train", "val", "test") for i in sp[k]}
    cut1, cut2 = (pd.Timestamp(sp["boundaries"][k]) for k in ("train_end", "val_end"))
    split_of_ts = np.where(al["ts"] < cut1, "train", np.where(al["ts"] < cut2, "val", "test"))
    assert (al["label_incident_id"].map(side).to_numpy() == split_of_ts).all()


def test_plan_is_deterministic_under_seed():
    a, b, c = build_dataset(5, 42), build_dataset(5, 42), build_dataset(5, 43)
    pd.testing.assert_frame_equal(a.alarms, b.alarms)
    pd.testing.assert_frame_equal(a.incidents_frame(), b.incidents_frame())
    assert not a.alarms.equals(c.alarms)
    (_, ka), (_, kb) = next(a.kpi_chunks()), next(b.kpi_chunks())
    np.testing.assert_array_equal(ka, kb)  # NaN (missing samples) compare equal


def test_seeded_db_matches_fresh_plan(small_db):
    s, _ = small_db
    db = pd.read_sql("SELECT * FROM alarms ORDER BY alarm_id", get_engine(s), parse_dates=["ts"])
    plan = build_dataset(days=2, seed=42).alarms.reset_index(drop=True)
    assert (db["alarm_id"].to_numpy() == plan["alarm_id"].to_numpy()).all()
    assert (db["raw_text"].to_numpy() == plan["raw_text"].to_numpy()).all()
    assert (db["label_is_true"].astype(bool).to_numpy() == plan["label_is_true"].to_numpy()).all()
