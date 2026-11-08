"""Update the original Figure 4/5 data views from saved current-model summaries."""
from pathlib import Path
import json
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("workdir", type=Path)
args = parser.parse_args()
ROOT = args.workdir
OUT = ROOT / "Source_Data"
PAIRED = 'Inputs/subsets/paired_per_seed.tsv'
EVAL = 'Inputs/subsets/'
MODS = ['rna', 'cnv', 'protein', 'methylation', 'he']
METRICS = ['pooled_c_index', 'within_cancer_c_index']

def read(path):
    return pd.read_csv(ROOT / path, sep='\t')

def summary(data, keys, columns):
    group = data.groupby(keys, sort=False)[columns]
    return pd.concat([group.mean().add_suffix('_mean'), group.std().add_suffix('_sd'),
                      group.size().rename('n_seeds')], axis=1).reset_index()

def save(figure, tables, sources):
    dest = OUT / figure
    dest.mkdir(parents=True, exist_ok=True)
    tables['provenance'] = pd.DataFrame(sources, columns=['source_file','use'])
    workbook = dest / f'Source_Data_{figure}.xlsx'
    with pd.ExcelWriter(workbook, engine='openpyxl') as writer:
        for name, table in tables.items():
            table.to_csv(dest / f'{name}.tsv', sep='\t', index=False, float_format='%.17g')
            table.to_excel(writer, sheet_name=name[:31], index=False)
    checks = []
    for name, expected in tables.items():
        actual = pd.read_excel(workbook, sheet_name=name[:31])
        assert actual.shape == expected.shape and list(actual) == list(expected)
        for col in expected.select_dtypes(include='number'):
            np.testing.assert_allclose(actual[col], expected[col], rtol=1e-12, atol=1e-12)
        checks.append({'table':name,'rows':len(expected),'excel_readback':'PASS'})
    (dest / 'source_readback.json').write_text(json.dumps(checks, indent=2)+'\n')
    print(figure, [(key,len(value)) for key,value in tables.items()])

paired = read(PAIRED)
paired = paired.loc[~paired.model.str.startswith('adapted_')].copy()
assert len(paired) == 9*31*5
assert not paired.duplicated(['model','seed','source_subset']).any()
for metric in METRICS:
    np.testing.assert_allclose(paired['delta_'+metric],
        paired['omnit2i_'+metric]-paired['baseline_'+metric],atol=1e-14)
omni = paired[['seed','source_subset','observed_count']+['omnit2i_'+m for m in METRICS]].drop_duplicates()
assert len(omni) == 31*5
omni = omni.rename(columns={'omnit2i_'+m:m for m in METRICS})
omni['model'] = 'omnit2i'
specialists = paired[['model','seed','source_subset','observed_count']+['baseline_'+m for m in METRICS]].rename(columns={'baseline_'+m:m for m in METRICS})
models = pd.concat([omni[specialists.columns], specialists],ignore_index=True)
subset = summary(paired,['model','source_subset','observed_count'],['delta_'+m for m in METRICS])
count_seed = models.groupby(['model','seed','observed_count'])[METRICS].mean().reset_index()
count_summary = summary(count_seed,['model','observed_count'],METRICS)
assert len(count_seed)==10*5*5 and len(count_summary)==10*5
save('Figure4', {'paired_per_seed':paired,'model_per_seed':models,
    'subset_differences':subset,'count_per_seed':count_seed,'count_summary':count_summary},
    [(PAIRED,'9 specialist families and current full-model predictions; 31 subsets, 5 seeds; 5605 patients, 1546 deaths')])

raw = read(EVAL+'selective/survival_per_seed.tsv')
features = read(EVAL+'summary/feature_per_seed.tsv')
assert len(raw)==3600 and len(features)==375
assert raw.n_patients.eq(5605).all() and raw.n_events.eq(1546).all()
keys = ['seed','source_subset','virtual_targets','observed_count','virtual_count','final_count']
matched = raw[keys].drop_duplicates().copy()
assert len(matched)==900
matched['measured_subset'] = matched.apply(lambda r: '+'.join(m for m in MODS
    if m in set(r.source_subset.split('+')) | set(r.virtual_targets.split('+'))),axis=1)
reference = omni.set_index(['seed','source_subset'])
for row in matched.itertuples():
    assert not set(row.source_subset.split('+')) & set(row.virtual_targets.split('+'))
    assert len(row.measured_subset.split('+')) == row.final_count
for metric in METRICS:
    wide = raw.pivot(index=keys,columns='condition',values=metric)
    for condition in ['source_only','virtual','cancer_mean']:
        matched = matched.merge(wide[condition].rename(condition+'_'+metric).reset_index(),on=keys,validate='one_to_one')
    matched['measured_matched_'+metric] = [reference.loc[(r.seed,r.measured_subset),metric] for r in matched.itertuples()]
    np.testing.assert_allclose(matched['source_only_'+metric],
        [reference.loc[(r.seed,r.source_subset),metric] for r in matched.itertuples()],atol=1e-12)
    for prefix,left,right in [('virtual_gain','virtual','source_only'),('measured_gain','measured_matched','source_only')]:
        matched[prefix+'_'+metric] = matched[left+'_'+metric]-matched[right+'_'+metric]
value_cols = [c for c in matched if any(c.endswith(m) for m in METRICS)]
routes = summary(matched,keys[1:]+['measured_subset'],value_cols)
assert len(routes)==180 and routes.n_seeds.eq(5).all()
single = matched[matched.virtual_count.eq(1)].rename(columns={'virtual_targets':'target'})
fcols = ['feature_pearson_mean_fold_weighted','within_cancer_feature_pearson_mean_fold_weighted']
joined = single.merge(features[['seed','source_subset','target']+fcols],on=['seed','source_subset','target'],validate='one_to_one')
assert len(joined)==375
target_seed = joined.groupby(['seed','target'])[value_cols+fcols].mean().reset_index()
target_summary = summary(target_seed,['target'],value_cols+fcols)
overview=[]
for metric in METRICS:
    virtual=routes['virtual_gain_'+metric+'_mean']; measured=routes['measured_gain_'+metric+'_mean']
    assert measured.mean()>0
    overview.append(dict(metric=metric,routes=180,virtual_positive=int((virtual>0).sum()),
        virtual_gain_mean=virtual.mean(),matched_measured_gain_mean=measured.mean(),
        ratio_of_mean_gains=virtual.mean()/measured.mean()))
    saved=read(EVAL+'selective/survival_deltas_summary.tsv')
    replay=routes.merge(saved,on=['source_subset','virtual_targets'],validate='one_to_one')
    np.testing.assert_allclose(replay['virtual_gain_'+metric+'_mean'],replay['delta_virtual_minus_source_only_'+metric+'_mean'],atol=1e-14)
save('Figure5',{'matched_per_seed':matched,'matched_routes':routes,'target_per_seed':target_seed,
    'target_summary':target_summary,'reconstruction_per_seed':features,'overview':pd.DataFrame(overview)},
    [(PAIRED,'Source union target measured reference; never substitute the all-five measured condition'),
     (EVAL+'selective/survival_per_seed.tsv','Source-only and source plus predicted target risk C-indices'),
     (EVAL+'summary/feature_per_seed.tsv','75 single-target reconstruction routes; fold-weighted feature correlations'),
     (EVAL+'selective/survival_deltas_summary.tsv','Independent readback of all 180 virtual gains')])
print(pd.DataFrame(overview).to_string(index=False))

# The supporting matrices are alternate views of the same verified current data.
heatmap_models = summary(models, ['model','source_subset','observed_count'], METRICS)
save('ExtendedData2', {'model_per_seed':models,'model_summary':heatmap_models},
    [(PAIRED, 'Same 10 models and 31 complete-case subsets as current Figure 4')])
route_views=[]
for metric, label in zip(METRICS, ['All patients','Within cancer']):
    view=routes[['source_subset','virtual_targets','observed_count','virtual_count']].copy()
    view['evaluation']=label
    view['virtual_gain']=routes['virtual_gain_'+metric+'_mean']
    route_views.append(view)
route_heatmap=pd.concat(route_views,ignore_index=True)
assert len(route_heatmap)==360
save('ExtendedData3', {'route_gains':route_heatmap,'matched_per_seed':matched},
    [(EVAL+'selective/survival_per_seed.tsv', 'Same 180 predicted-minus-source gains as current Figure 5')])
