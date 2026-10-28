from pathlib import Path
import json

import numpy as np
import pandas as pd


SOURCE = Path('/Users/yilingz/Downloads/train_data.csv')
OUT = Path('analysis_outputs')
OUT.mkdir(exist_ok=True)

df = pd.read_csv(SOURCE)
target = 'is_fraud'

# Transaction-consistency features are descriptive checks, not proposed production inputs.
df['sender_debit'] = df['sender_balance_before'] - df['sender_balance_after']
df['recipient_credit'] = df['recipient_balance_after'] - df['recipient_balance_before']
df['sender_amount_gap'] = (df['sender_debit'] - df['transaction_amount']).abs()
df['recipient_amount_gap'] = (df['recipient_credit'] - df['transaction_amount']).abs()
df['sender_emptied'] = (
    (df['sender_balance_before'] > 0) & (df['sender_balance_after'] == 0)
)
df['recipient_new'] = df['recipient_balance_before'] == 0
df['sender_balance_zero'] = df['sender_balance_before'] == 0
df['exact_sender_amount'] = df['sender_amount_gap'] <= 0.01
df['exact_recipient_amount'] = df['recipient_amount_gap'] <= 0.01
df['recipient_kind'] = df['recipient_id'].str[0].fillna('?')

def pct(x):
    return float(x) if pd.notna(x) else None

def group_summary(column):
    return (df.groupby(column, observed=False)[target]
            .agg(transactions='size', fraud_count='sum', fraud_rate='mean')
            .reset_index()
            .sort_values('fraud_rate', ascending=False))

by_type = group_summary('transaction_type')
by_recipient_kind = group_summary('recipient_kind')
feature_rates = []
for feature in ['sender_emptied', 'recipient_new', 'sender_balance_zero',
                'exact_sender_amount', 'exact_recipient_amount', 'is_flagged_fraud']:
    table = group_summary(feature)
    for _, row in table.iterrows():
        feature_rates.append({
            'feature': feature, 'value': str(row[feature]),
            'transactions': int(row['transactions']), 'fraud_count': int(row['fraud_count']),
            'fraud_rate': float(row['fraud_rate']),
        })
feature_rates = pd.DataFrame(feature_rates)

amount_quantiles = (df.groupby(target)['transaction_amount']
                    .quantile([0, .01, .05, .25, .5, .75, .95, .99, 1])
                    .unstack().reset_index())
amount_quantiles.columns = ['is_fraud', 'min', 'p01', 'p05', 'p25', 'median', 'p75', 'p95', 'p99', 'max']

fraud_by_time = (df.groupby('time_step')[target]
                 .agg(transactions='size', fraud_count='sum', fraud_rate='mean')
                 .reset_index())

# Capacity calculation: rank only with raw amount, a transparent EDA benchmark—not a model.
capacity_n = max(1, int(np.ceil(len(df) * .01)))
def capacity_result(frame, label):
    selected = frame.nlargest(min(capacity_n, len(frame)), 'transaction_amount')
    return {
        'ranking': label,
        'reviewed': int(len(selected)),
        'review_share': float(len(selected) / len(df)),
        'fraud_found': int(selected[target].sum()),
        'fraud_recall': float(selected[target].sum() / df[target].sum()),
        'precision': float(selected[target].mean()),
        'amount_cutoff': float(selected['transaction_amount'].min()),
    }
capacity = pd.DataFrame([
    capacity_result(df, 'All transactions: highest amount'),
    capacity_result(df[df['transaction_type'].isin(['TRANSFER', 'CASH_OUT'])],
                    'TRANSFER/CASH_OUT only: highest amount'),
])

# Candidate segments for a non-model shortlist. Report all segments covering <=2% to show concentration.
segments = []
for tx_type in sorted(df['transaction_type'].unique()):
    for emptied in [False, True]:
        for rnew in [False, True]:
            s = df[(df.transaction_type == tx_type) & (df.sender_emptied == emptied) & (df.recipient_new == rnew)]
            if len(s):
                segments.append({
                    'transaction_type': tx_type, 'sender_emptied': emptied, 'recipient_new': rnew,
                    'transactions': len(s), 'share': len(s)/len(df), 'fraud_count': int(s[target].sum()),
                    'fraud_rate': s[target].mean(), 'fraud_recall': s[target].sum()/df[target].sum(),
                })
segments = pd.DataFrame(segments).sort_values(['fraud_rate', 'fraud_count'], ascending=False)

summary = {
    'rows': int(len(df)),
    'columns': int(11),
    'date_range': {'time_step_min': int(df.time_step.min()), 'time_step_max': int(df.time_step.max()),
                   'unique_time_steps': int(df.time_step.nunique())},
    'fraud_count': int(df[target].sum()),
    'fraud_rate': float(df[target].mean()),
    'flagged_count': int(df.is_flagged_fraud.sum()),
    'flagged_fraud_count': int(df.loc[df.is_flagged_fraud == 1, target].sum()),
    'missing_values': {k: int(v) for k, v in df.iloc[:, :11].isna().sum().items()},
    'duplicate_rows': int(df.iloc[:, :11].duplicated().sum()),
    'unique_senders': int(df.sender_id.nunique()),
    'unique_recipients': int(df.recipient_id.nunique()),
}

# Save compact, auditable tables.
by_type.to_csv(OUT / 'fraud_by_transaction_type.csv', index=False)
by_recipient_kind.to_csv(OUT / 'fraud_by_recipient_prefix.csv', index=False)
feature_rates.to_csv(OUT / 'derived_feature_rates.csv', index=False)
amount_quantiles.to_csv(OUT / 'amount_quantiles_by_fraud.csv', index=False)
fraud_by_time.to_csv(OUT / 'fraud_by_time_step.csv', index=False)
capacity.to_csv(OUT / 'one_percent_amount_ranking.csv', index=False)
segments.to_csv(OUT / 'candidate_risk_segments.csv', index=False)
with open(OUT / 'eda_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
print('\nBY TYPE\n', by_type.to_string(index=False))
print('\nFEATURE RATES\n', feature_rates.to_string(index=False))
print('\nAMOUNT QUANTILES\n', amount_quantiles.to_string(index=False))
print('\n1% CAPACITY\n', capacity.to_string(index=False))
print('\nTOP SEGMENTS\n', segments.head(12).to_string(index=False))
