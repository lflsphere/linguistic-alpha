#!/usr/bin/env python3
"""Prepare or fit a selectable Model A financial baseline; all outputs are local."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from linguistic_alpha.config import REPO_ROOT, load_config
from linguistic_alpha.controls.financial_baselines import CORE_COLUMNS, MOMENTUM_COLUMN
from linguistic_alpha.models.financial_baselines import prepare_panel
from run_model_a_dashboard import pipeline
from sklearn.metrics import mean_squared_error, r2_score


DEFAULT_CONFIG = REPO_ROOT/"configs/model_a_financial.yaml"


def fit_spec(panel,settings,spec,out):
    if panel.keydevid.isna().any() or panel.keydevid.duplicated().any():
        raise ValueError("Prepared input must have unique nonmissing event IDs")
    fields = CORE_COLUMNS.copy()
    if settings["specifications"][spec]["add_short_momentum"]:
        fields.append(MOMENTUM_COLUMN)
    rows, predictions = [], []
    for target in settings["target_windows"]:
        eligible = panel["eligible_"+target].fillna(False)
        train = panel[eligible & panel.split.eq("train")].copy()
        test = panel[eligible & panel.split.eq("test")].copy()
        if min(len(train),len(test)) < 2:
            raise ValueError(f"Empty or insufficient cohort for {target}")
        train[fields] = train[fields].astype(float)
        test[fields] = test[fields].astype(float)
        y = test[target].astype(float)
        for model in settings["models"]:
            estimator = pipeline(model,train,int(settings["seed"]),controls=fields)
            estimator.fit(train,train[target].astype(float))
            pred = estimator.predict(test)
            series = pd.Series(pred,index=y.index)
            rows.append({"specification":spec,"target":target,"model":model,
                         "train_n":len(train),"test_n":len(test),"r2":r2_score(y,pred),
                         "rank_ic":y.corr(series,method="spearman"),
                         "rmse":mean_squared_error(y,pred)**.5,"features":",".join(fields)})
            predictions.append(pd.DataFrame({"specification":spec,"target":target,"model":model,
                "keydevid":test.keydevid.to_numpy(),"actual":y.to_numpy(),"prediction":pred}))
        print(f"{spec} {target}: {len(train):,} train / {len(test):,} test",flush=True)
    pd.DataFrame(rows).to_csv(out/"metrics.csv",index=False)
    pd.concat(predictions,ignore_index=True).to_parquet(out/"predictions.parquet",index=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=DEFAULT_CONFIG)
    sub=parser.add_subparsers(dest="command",required=True)
    prepare=sub.add_parser("prepare",help="Prepare local panel without reading transcripts")
    prepare.add_argument("--raw",type=Path,required=True)
    prepare.add_argument("--baseline",type=Path,required=True)
    prepare.add_argument("--schedule",type=Path)
    prepare.add_argument("--out",type=Path,required=True)
    fit=sub.add_parser("fit",help="Fit one of the configured specifications")
    fit.add_argument("--panel",type=Path,required=True)
    fit.add_argument("--spec",required=True)
    fit.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    settings=load_config(args.config)["financial_baselines"]
    if args.command=="prepare":
        panel=prepare_panel(args.raw,args.baseline,settings,args.schedule)
        args.out.mkdir(parents=True,exist_ok=False)
        panel.to_parquet(args.out/"financial_panel.parquet",index=False)
        print(f"Wrote {len(panel):,} events to {args.out}",flush=True)
    else:
        if args.spec not in settings["specifications"]:
            parser.error(f"Unknown specification: {args.spec}")
        panel=pd.read_parquet(args.panel)
        args.out.mkdir(parents=True,exist_ok=False)
        fit_spec(panel,settings,args.spec,args.out)


if __name__=="__main__":
    main()
