#!/usr/bin/env python3
"""Compare the rebuilt baselines with prior local, authorized experiment artifacts."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from linguistic_alpha.controls.financial_baselines import CORE_COLUMNS, MOMENTUM_COLUMN


def compare(prepared, a2, enhanced, alignment, validation, ablation):
    rebuilt = pd.read_parquet(prepared).set_index("keydevid",verify_integrity=True)
    old = pd.read_parquet(alignment/"prepared_financial_panel.parquet").set_index("keydevid",verify_integrity=True)
    validated = pd.read_parquet(ablation/"paired_predictions.parquet")
    reference = pd.read_parquet(alignment/"paired_predictions.parquet")
    ledger = pd.read_parquet(alignment/"model_event_membership.parquet")
    if set(old.index)!=set(rebuilt.index):
        raise AssertionError("The rebuilt event universe differs")
    old = old.loc[rebuilt.index].copy()
    momentum = pd.read_parquet(validation/"validated_panel.parquet",columns=["keydevid",MOMENTUM_COLUMN])
    old[MOMENTUM_COLUMN] = momentum.set_index("keydevid",verify_integrity=True).loc[rebuilt.index,MOMENTUM_COLUMN]
    for left,right in (("day0","conservative_day0"),("original_day0","original_day0"),
                       ("signal_immediate_utc","signal_immediate_utc"),("split","split")):
        pd.testing.assert_series_equal(rebuilt[left],old[right],check_names=False,check_index_type=False)
    for col in CORE_COLUMNS + [MOMENTUM_COLUMN]:
        np.testing.assert_allclose(rebuilt[col].to_numpy(float,na_value=np.nan),old[col].to_numpy(float,na_value=np.nan),equal_nan=True,rtol=1e-10,atol=1e-12,
                                  err_msg=f"Feature changed: {col}")
    results=[]
    for name,folder,expected,ref in (
        ("a2",a2,reference.query("sample == 'matched'") if "sample" in reference else reference,"A2"),
        ("a2_short_momentum",enhanced,validated,"M1")
    ):
        variant_key="variant" if name=="a2" else "spec"
        prediction=pd.read_parquet(folder/"predictions.parquet")
        metrics=pd.read_csv(folder/"metrics.csv")
        for row in metrics.itertuples(index=False):
            target,model=row.target,row.model
            controls = CORE_COLUMNS + ([MOMENTUM_COLUMN] if name == "a2_short_momentum" else [])
            if row.features.split(",") != controls:
                raise AssertionError("Unexpected fitted feature list")
            np.testing.assert_allclose(rebuilt[target],old["B_"+target],equal_nan=True,rtol=1e-10,atol=1e-12)
            members=ledger[(ledger["sample"]=="matched") & (ledger.target==target) & (ledger.variant=="A2")]
            keep=rebuilt["eligible_"+target]
            for split in ("train","test"):
                ids=set(rebuilt.index[keep & rebuilt.split.eq(split)])
                if ids!=set(members.loc[members.split.eq(split),"keydevid"]):
                    raise AssertionError(f"Cohort changed: {target}, {split}")
                if len(ids)!=getattr(row,split+"_n"):
                    raise AssertionError("Reported cohort count differs from the frozen IDs")
            old_pred=expected[(expected[variant_key]==ref) & (expected.model==model) & (expected.target==target)]
            got=prediction[(prediction.model==model)&(prediction.target==target)].set_index("keydevid",verify_integrity=True)
            if set(old_pred.keydevid)!=set(got.index):
                raise AssertionError("Forecast event IDs differ")
            old_pred=old_pred.set_index("keydevid",verify_integrity=True).loc[got.index]
            np.testing.assert_allclose(got.actual,old_pred.actual,rtol=1e-10,atol=1e-12)
            np.testing.assert_allclose(got.prediction,old_pred.prediction,rtol=1e-9,atol=1e-11)
            source = pd.read_csv(alignment/"model_metrics.csv" if name=="a2" else ablation/"factorial_ablation_metrics.csv")
            if name == "a2":
                source = source[source["sample"].eq("matched")]
            previous = source[(source[variant_key]==ref)&(source.model==model)&(source.target==target)].iloc[0]
            np.testing.assert_allclose([row.r2,row.rank_ic,row.rmse],
                                       previous[["r2","rank_ic","rmse"]].to_numpy(float),rtol=1e-9,atol=1e-11)
            results.append({"specification":name,"target":target,"model":model,
                "train_n":row.train_n,"test_n":row.test_n,"r2":row.r2,"rank_ic":row.rank_ic,"rmse":row.rmse,
                "max_abs_prediction_delta":(got.prediction-old_pred.prediction).abs().max()})
    return pd.DataFrame(results)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ("prepared","a2","enhanced","alignment","validation","ablation","out"):
        ap.add_argument("--"+name,type=Path,required=True)
    args=ap.parse_args()
    result=compare(args.prepared,args.a2,args.enhanced,args.alignment,args.validation,args.ablation)
    with args.out.open("x",encoding="utf-8",newline="") as f:
        result.to_csv(f,index=False)
    print(result[result.target.eq("car_2_20")].to_string(index=False))


if __name__=="__main__":
    main()
