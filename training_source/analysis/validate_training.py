"""Validate a completed local training run without refitting or tuning."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from train_baseline_xgboost import BASE, DATA, baseline_predictions, choose, digest, evaluate, json_write


def validate(output):
    summary=json.loads((output/'summary.json').read_text(encoding='utf-8'))
    before=json.loads((output/'config_before_fit.json').read_text(encoding='utf-8'))
    decision=json.loads((output/'selection_before_test.json').read_text(encoding='utf-8'))
    provenance=json.loads((output/'input_provenance.json').read_text(encoding='utf-8'))
    assert choose(decision['validation_candidates'])==summary['selected_xgboost']
    assert decision['selected_xgboost']==summary['selected_xgboost']
    assert before['features']==summary['config']['features']
    assert not any(c.startswith(('audit_','review_')) for c in before['features'])
    assert before['label'] not in before['features']
    assert digest(DATA/'selection_manifest.json')==before['source_manifest_sha256']
    for row in provenance:
        assert digest(Path(row['path']))==row['sha256'],row['path']
    test_parts=[pd.read_parquet(row['path']) for row in provenance if row['split']=='test']
    test=pd.concat(test_parts).sort_index(kind='stable')
    stored=pd.read_parquet(output/'test_predictions_local.parquet')
    assert stored.index.equals(test.index)
    assert stored.deployment_id.equals(test.deployment_id)
    assert stored[before['label']].equals(test[before['label']])
    model=XGBRegressor()
    model.load_model(output/'xgboost_selected.ubj')
    predictions=baseline_predictions(test)
    predictions['xgboost_selected']=np.maximum(model.predict(test[before['features']]),0)
    for name,pred in predictions.items():
        np.testing.assert_allclose(stored[f'prediction_{name}_cm'],pred,rtol=1e-6,atol=1e-6)
        actual=evaluate(test[before['label']],pred,test.depth_lag_0_cm)
        expected=next(r for r in summary['test_metrics'] if r['model']==name)
        for metric,value in actual.items():
            if value is None:
                assert expected[metric] is None,(name,metric)
            else:
                np.testing.assert_allclose(value,expected[metric],rtol=1e-8,atol=1e-8)
    result={'status':'PASS','test_rows_repredicted':len(test),'source_files_sha256_checked':len(provenance),
        'selected_xgboost':summary['selected_xgboost'],
        'checks':['validation selection reproduces','all source checksums match','feature whitelist excludes labels/metadata',
                  'stored model reproduces predictions','baselines reproduce','all test metrics reproduce'],
        'limits':'Structural/reproducibility checks only, not proof of causal filtering, true flood labels or operational safety.'}
    json_write(output/'artifact_validation.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('run_directory',type=Path)
    args=parser.parse_args()
    validate(args.run_directory)
