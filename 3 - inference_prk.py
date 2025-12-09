import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
import openai
from openai import AzureOpenAI, OpenAI
import pandas as pd
import numpy as np
import re
from modules import *
from tqdm import tqdm
from pathlib import Path
import sys, traceback

from utils import query_gpt_openai, get_prompt_for_inference_in_context, random_selection_GRDP
import argparse
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------
# Args
# ---------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--ccode', required=True, help='ccode', default='')
parser.add_argument('--ablation', required=False, help='ablation', default='')
parser.add_argument('--target_var', required=False, help='testing variable')
parser.add_argument('--model', required=True, help='llm model')
parser.add_argument('--timeline', required=False, help='timeline (year) using')
parser.add_argument('--csv', required=True, default=None,
                    help='explicit path to paragraph CSV; ')
# parser.add_argument('--label_csv', required=True,
#                     help='Path to label CSV with columns [area_id, target_var]')

args = parser.parse_args()

if args.ablation:
    ablation = '_' + args.ablation
else:
    ablation = ''

target_var = args.target_var
ccode = args.ccode
model = args.model
# timeline = str(args.timeline)
timeline = "2022 to 2025"

if model == "QwQ":
    model = "Qwen/QwQ-32B"

model_tag = model.split('/')[-1]


file_path_var = target_var
col_var = target_var
adm_offset = ''
iter_num = 3
iter_init_offset = 2
random_seed = list(range(iter_init_offset, iter_init_offset + iter_num))

# ---------------------------------------------------------
# LLM client setup
# ---------------------------------------------------------
endpoint = 
subscription_key = 
api_version = "2025-04-01-preview"
azure_models = ['Llama-3.3-70B-Instruct','o4-mini' ]
openai_latest_models = ['gpt-5.1']
hf_models = ["meta-llama/Llama-3.3-70B-Instruct", "Qwen/QwQ-32B"]
model_tag = model.split('/')[-1]
if model in azure_models:
    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=subscription_key,
    )
elif model in openai_latest_models:
    client = OpenAI(
        # base_url="https://router.huggingface.co/v1",
        api_key=
    )
else:
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=
    )

prompt_version = '2'

# ---------------------------------------------------------
# Target description
# ---------------------------------------------------------
if target_var == 'GRDP':
    target_var_desc = 'regional GDP (in Million USD)'
elif target_var == 'population':
    target_var_desc = 'the population'
else:
    target_var_desc = target_var

# number of few-shot examples to draw per iteration
shot = 4

# ---------------------------------------------------------
# Model family helper
# ---------------------------------------------------------
def _family(name: str) -> str:
    n = name.lower()
    if 'llama' in n:
        return 'llama'
    if 'qwq' in n:
        return 'qwq'
    return 'gpt'

QWQ_COT = True

fam = _family(model)
if fam == 'llama':
    fewshot_target_question = (
        f"Infer {target_var_desc} from given location's description by utilizing multiple data aspects of given data "
        f"and including any additional information you have about the region on {timeline}. "
        f"Following are ground truths of other regions of {ccode} (year {timeline}). "
        f"Note that data and answer of Cheongjin, Pyeongyang, Rason and Shiuiju are estimates of year 2020. "
        f"Note that 2025 night-lights data is not yet available; the night-lights shown for 2025 regions are simply the 2024 values inserted in their place. "

        f"Do not show your reasoning process. Answer the numeric score only."
    )
elif fam == 'gpt':
    fewshot_target_question = (
        f"Infer {target_var_desc} from given location's description by utilizing multiple data aspects of given data "
        f"and including any additional information you have about the region on {timeline}. "
        f"Following are ground truths of other regions of {ccode} (year {timeline}). "
                f"Note that data and answer of Cheongjin, Pyeongyang, Rason and Shiuiju are estimates of year 2020. "
        f"Note that 2025 night-lights data is not yet available; the night-lights shown for 2025 regions are simply the 2024 values inserted in their place. "
        f"Show your reasoning process."
    )
else:  # qwq
    fewshot_target_question = (
        f"Infer {target_var_desc} from given location's description by utilizing multiple data aspects of given data "
        f"and including any additional information you have about the region on {timeline}. "
        f"Following are ground truths of other regions of {ccode} (year {timeline}). "
                f"Note that data and answer of Cheongjin, Pyeongyang, Rason and Shiuiju are estimates of year 2020. "
        f"Note that 2025 night-lights data is not yet available; the night-lights shown for 2025 regions are simply the 2024 values inserted in their place. "
        + ("Show your reasoning process." if QWQ_COT else "Answer the numeric score only.")
    )

# ---------------------------------------------------------
# Load paragraph CSV
# ---------------------------------------------------------
if args.csv:
    csv_path = Path(args.csv)
else:
    csv_path = Path(f"./extracted_paragraphs/{ccode}/{ccode}_{model_tag}_{target_var}_paragraph_output.csv")
base_name = csv_path.stem 
try:
    df = pd.read_csv(csv_path, dtype={'area_id': str})
except Exception as e:
    print(f"[fatal] failed to read CSV: {csv_path} :: {e}", file=sys.stderr)
    sys.exit(2)

# ---------------------------------------------------------
# Load label CSV and build candidate pool
# ---------------------------------------------------------
# label_csv_path = Path(args.label_csv)
label_df = df[['area_id']].copy()
label_df['score'] = -1
label_df['GRDP'] = -1
label_df.iloc[:4,-1] = [869,6470, 359, 1027]
print(label_df)
# try:
#     label_df = pd.read_csv(label_csv_path, dtype={'area_id': str})
# except Exception as e:
#     print(f"[fatal] failed to read label CSV: {label_csv_path} :: {e}", file=sys.stderr)
#     sys.exit(2)


# Filter to same year (suffix _{timeline})
# label_df_year = label_df[label_df['area_id'].str.endswith(f"_{timeline}")].copy()

# Keep only area_ids that also exist in df
# label_df_year = label_df_year[label_df_year['area_id'].isin(df['area_id'])].reset_index(drop=True)

# print(f"[info] few-shot candidate pool for year {timeline}: {len(label_df_year)} regions")

# Overall result container
# result_df_overall = pd.DataFrame(
#     columns=[
#         'spearman_corr_mean', 'spearman_corr_std',
#         'pearson_corr_mean', 'pearson_corr_std',
#         'r2_mean', 'r2_std',
#         'r2_intercept_mean', 'r2_intercept_std',
#         'test_rmse_mean', 'test_rmse_std'
#     ],
#     index=[ccode]
# )

# ---------------------------------------------------------
# Main loop: for each iteration, sample random few-shots from label_df_year
# ---------------------------------------------------------
COT_reasonings = pd.DataFrame(columns=["desc"])

results_dir = Path(f'PRK_20251127/results')
results_dir.mkdir(parents=True, exist_ok=True)
cot_base_dir = Path(f"PRK_20251127/COT_reasoning")
cot_base_dir.mkdir(parents=True, exist_ok=True)
plot_base_dir = Path(f'PRK_20251127/results_plot')
plot_base_dir.mkdir(parents=True, exist_ok=True)
excel_base_dir = Path(f'PRK_20251127/results_excel')
excel_base_dir.mkdir(parents=True, exist_ok=True)

for iteration in range(iter_num):
    out_path = results_dir / f"{base_name}_iter_num_{iteration}.csv"
    print(results_dir)
    print(out_path)
    # if out_path.exists():
    #     print(f"[info] skip iteration {iteration}: {out_path} already exists")
    #     continue


    fixed_ids = label_df['area_id'].astype(str).head(4).tolist()
    shot = len(fixed_ids)          # few-shot 개수 4로 고정
    temp_random_ids = fixed_ids
    random_ids = [str(x) for x in temp_random_ids]

    train_mask = label_df['area_id'].isin(random_ids)
    # print(train_mask.value_counts())
    # assert(0)
    train_df = label_df[train_mask].reset_index(drop=True)

    # Test set: all other labeled regions of that year
    test_df = label_df[~train_mask].reset_index(drop=True)

    X_train = train_df[['area_id']]
    y_train = train_df[[target_var]].rename(columns={target_var: col_var})
    X_test = test_df[['area_id']]
    y_test = test_df[[target_var]].rename(columns={target_var: col_var})

    df_record = pd.DataFrame(X_test.copy())
    df_record['score'] = [-1 for _ in range(len(df_record))]
    df_record['ground_truth'] = y_test[col_var].to_numpy()
    for i in tqdm(range(len(X_test)), desc=f"iter {iteration}"):
        inference_prompt = get_prompt_for_inference_in_context(
            df, df_record, i, X_train, y_train, X_test, fewshot_target_question
        )
        # print(inference_prompt)
        # assert(0)
        model_output = query_gpt_openai(
            [inference_prompt], client, model=model, temperature=0.5,
            max_completion_tokens=25000, max_try_num=10
        )[0]
        model_output = model_output.replace(',', '')

        COT_reasonings.loc[i] = model_output
        cot_path = cot_base_dir / f"{base_name}_{iteration}.csv"
        COT_reasonings.to_csv(cot_path, index=False)

        try:
            cleaned = re.sub(r"(?<=\d)\s+(?=\d)", "", model_output)

            # number = 
            answer = float(re.findall(r"[-+]?\d*\.\d+|\d+", cleaned)[-1])
        except Exception:
            answer = -1

        df_record.loc[i, 'score'] = answer

    # attach ground truth for the same year
    

        df_record.to_csv(out_path, index=False, header=True)
    print(f"[info] saved {out_path}")

# ---------------------------------------------------------
# Metrics over iterations (Spearman, Pearson, R², etc.)
# ---------------------------------------------------------
# from sklearn.metrics import r2_score, mean_squared_error
# from sklearn.linear_model import LinearRegression
# import matplotlib.pyplot as plt

# spearman_list = []
# pearson_list = []
# r2_list = []
# r2i_list = []
# rmse_list = []

# metrics_per_year = {}

# base_name = f"{args.ccode}_{args.model}_{args.var}_{timeline}"

# for iteration in range(args.iter_num):

#     result_path = results_dir / f"{base_name}_iter_{iteration}.csv"
#     if not result_path.exists():
#         continue

#     df_record = pd.read_csv(result_path)

#     if 'ground_truth' not in df_record.columns:
#         continue

#     sp = df_record[['score','ground_truth']].corr(method='spearman').iloc[0,1]
#     pe = df_record[['score','ground_truth']].corr(method='pearson').iloc[0,1]
#     r2 = r2_score(df_record['ground_truth'], df_record['score'])

#     ols = LinearRegression(fit_intercept=True)
#     ols.fit(df_record[['score']], df_record[['ground_truth']])
#     y_pred = ols.predict(df_record[['score']])
#     r2i = r2_score(df_record['ground_truth'], y_pred)

#     rmse = np.sqrt(mean_squared_error(df_record['score'], df_record['ground_truth']))

#     spearman_list.append(sp)
#     pearson_list.append(pe)
#     r2_list.append(r2)
#     r2i_list.append(r2i)
#     rmse_list.append(rmse)

#     df_record['year'] = df_record['area_id'].astype(str).str.extract(r'(\d{4})$')

#     for year, group in df_record.groupby('year'):
#         if year not in metrics_per_year:
#             metrics_per_year[year] = {'spearman':[], 'pearson':[], 'r2':[], 'r2i':[], 'rmse':[]}

#         sp_y = group[['score','ground_truth']].corr(method='spearman').iloc[0,1]
#         pe_y = group[['score','ground_truth']].corr(method='pearson').iloc[0,1]
#         r2_y = r2_score(group['ground_truth'], group['score'])

#         ols_y = LinearRegression(fit_intercept=True)
#         ols_y.fit(group[['score']], group[['ground_truth']])
#         y_pred_y = ols_y.predict(group[['score']])
#         r2i_y = r2_score(group['ground_truth'], y_pred_y)

#         rmse_y = np.sqrt(mean_squared_error(group['score'], group['ground_truth']))

#         metrics_per_year[year]['spearman'].append(sp_y)
#         metrics_per_year[year]['pearson'].append(pe_y)
#         metrics_per_year[year]['r2'].append(r2_y)
#         metrics_per_year[year]['r2i'].append(r2i_y)
#         metrics_per_year[year]['rmse'].append(rmse_y)

# # ---------------------------------------------------------
# # Save final CSV (combined overall + year rows)
# # ---------------------------------------------------------

# rows = []

# # Overall row
# rows.append({
#     'type': 'overall',
#     'year': 'ALL',
#     'spearman_mean': np.mean(spearman_list),
#     'spearman_std': np.std(spearman_list),
#     'pearson_mean': np.mean(pearson_list),
#     'pearson_std': np.std(pearson_list),
#     'r2_mean': np.mean(r2_list),
#     'r2_std': np.std(r2_list),
#     'r2i_mean': np.mean(r2i_list),
#     'r2i_std': np.std(r2i_list),
#     'rmse_mean': np.mean(rmse_list),
#     'rmse_std': np.std(rmse_list),
# })

# # Year rows
# for year in sorted(metrics_per_year.keys()):
#     d = metrics_per_year[year]
#     rows.append({
#         'type': 'year',
#         'year': year,
#         'spearman_mean': np.mean(d['spearman']),
#         'spearman_std': np.std(d['spearman']),
#         'pearson_mean': np.mean(d['pearson']),
#         'pearson_std': np.std(d['pearson']),
#         'r2_mean': np.mean(d['r2']),
#         'r2_std': np.std(d['r2']),
#         'r2i_mean': np.mean(d['r2i']),
#         'r2i_std': np.std(d['r2i']),
#         'rmse_mean': np.mean(d['rmse']),
#         'rmse_std': np.std(d['rmse']),
#     })

# final_df = pd.DataFrame(rows)

# final_path = excel_base_dir / f"{base_name}_metrics.csv"
# final_df.to_csv(final_path, index=False)

# print(f"[info] saved final metrics CSV → {final_path}")