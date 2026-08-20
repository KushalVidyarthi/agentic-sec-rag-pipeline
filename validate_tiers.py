import warnings
warnings.filterwarnings('ignore')
from extract_financials import find_target_page, extract_financial_table
from agent_pipeline import _normalise_df, execute_tabular_query

page = find_target_page('apple_10k.pdf', 'CONSOLIDATED STATEMENTS OF OPERATIONS')
df = _normalise_df(extract_financial_table('apple_10k.pdf', page))

tier1_queries = [
    'What was net income in 2025?',
    'Did gross margin increase in 2025 vs 2024?',
    'What was total net sales in 2025 compared to 2024?',
    'Show me operating income for 2025',
]

tier2_query = 'What was the gross margin percentage for the Services segment in 2025?'

print('--- TIER 1 tests (expect: no ESCALATION message) ---')
for q in tier1_queries:
    r = execute_tabular_query(q, df)
    matched = r['matched']
    tier = 'T1' if matched not in ('none', 'llm_agent') else 'T2/T3'
    print('[' + tier + '] ' + q[:55])
    print('       matched=' + matched)
    print(r['answer'])
    print()

print('--- TIER 2 test (expect: ESCALATION message) ---')
r = execute_tabular_query(tier2_query, df)
print('matched =', r['matched'])
print(r['answer'][:300])
