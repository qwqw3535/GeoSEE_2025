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
parser.add_argument('--model', required=False, help='llm model')
parser.add_argument('--timeline', required=False, help='timeline (year) using')
parser.add_argument('--csv', required=False, default=None,
                    help='explicit path to paragraph CSV; ')
parser.add_argument('--label_csv', required=True,
                    help='Path to label CSV with columns [area_id, target_var]')

args = parser.parse_args()

if args.ablation:
    ablation = '_' + args.ablation
else:
    ablation = ''

target_var = args.target_var
ccode = args.ccode
model = args.model
timeline = str(args.timeline)

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
endpoint = ""
subscription_key = ""
api_version = "2025-04-01-preview"
azure_models = ['o4-mini','Llama-3.3-70B-Instruct' ]
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
        api_key=""
    )
else:
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=""
    )

prompt_version = '2'

# ---------------------------------------------------------
# Target description
# ---------------------------------------------------------
if target_var == 'GRDP':
    target_var_desc = 'regional GDP (in Billion KRW)'
elif target_var == 'population':
    target_var_desc = 'the population'
else:
    target_var_desc = target_var

# number of few-shot examples to draw per iteration
shot = 5

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
        # f"Answer the numeric score only."
        f"Do not show your reasoning process. Answer the numeric score only."
    )
elif fam == 'gpt':
    fewshot_target_question = (
        f"Infer {target_var_desc} from given location's description by utilizing multiple data aspects of given data "
        f"and including any additional information you have about the region on {timeline}. "
        f"Following are ground truths of other regions of {ccode} (year {timeline}). "
        f"Show your reasoning process."
    )
else:  # qwq
    fewshot_target_question = (
        f"Infer {target_var_desc} from given location's description by utilizing multiple data aspects of given data "
        f"and including any additional information you have about the region on {timeline}. "
        f"Following are ground truths of other regions of {ccode} (year {timeline}). "
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
label_csv_path = Path(args.label_csv)

try:
    label_df = pd.read_csv(label_csv_path, dtype={'area_id': str})
except Exception as e:
    print(f"[fatal] failed to read label CSV: {label_csv_path} :: {e}", file=sys.stderr)
    sys.exit(2)


# Filter to same year (suffix _{timeline})
# label_df_year = label_df[label_df['area_id'].str.endswith(f"_{timeline}")].copy()

# Keep only area_ids that also exist in df
# label_df_year = label_df_year[label_df_year['area_id'].isin(df['area_id'])].reset_index(drop=True)

# print(f"[info] few-shot candidate pool for year {timeline}: {len(label_df_year)} regions")

# Overall result container
result_df_overall = pd.DataFrame(
    columns=[
        'spearman_corr_mean', 'spearman_corr_std',
        'pearson_corr_mean', 'pearson_corr_std',
        'r2_mean', 'r2_std',
        'r2_intercept_mean', 'r2_intercept_std',
        'test_rmse_mean', 'test_rmse_std'
    ],
    index=[ccode]
)

# ---------------------------------------------------------
# Main loop: for each iteration, sample random few-shots from label_df_year
# ---------------------------------------------------------
COT_reasonings = pd.DataFrame(columns=["desc"])

results_dir = Path(f'test2/results')
results_dir.mkdir(parents=True, exist_ok=True)
cot_base_dir = Path(f"test2/COT_reasoning")
cot_base_dir.mkdir(parents=True, exist_ok=True)
plot_base_dir = Path(f'test2/results_plot')
plot_base_dir.mkdir(parents=True, exist_ok=True)
excel_base_dir = Path(f'test2/results_excel')
excel_base_dir.mkdir(parents=True, exist_ok=True)

for iteration in range(iter_num):
    out_path = results_dir / f"{base_name}_iter_num_{iteration}.csv"
    print(results_dir)
    print(out_path)
    # if out_path.exists():
    #     print(f"[info] skip iteration {iteration}: {out_path} already exists")
    #     continue

    # Sample few-shot regions randomly same year FROM THE LABELS
    rng_seed = random_seed[iteration]
    temp_random_ids = random_selection_GRDP(
        str(label_csv_path),
        n_shot=shot,
        seed=rng_seed,
        # timeline=timeline,   # ensure area_id ends with _{timeline}
    )
    random_ids = [str(x) for x in temp_random_ids]
    # print(type(random_ids[0]))
    # print(type(label_df['area_id'].iloc[0]))
    # Train set: rows in label_df_year whose area_id is selected
    # train_mask = label_df['area_id'].isin(['42810', '47850', '26170', '29170'])
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
spearman_corr_list = []
pearson_corr_list = []
r_2_list = []
r_2_intercept_list = []
test_rmse_list = []

# Note: len(X_train) === shot, but we use 'shot' consistently in filename
for iteration in range(iter_num):
    result_path = results_dir / f"{base_name}_iter_num_{iteration}.csv"
    if not result_path.exists():
        continue

    df_record = pd.read_csv(result_path)

    if 'ground_truth' not in df_record.columns:
        print(f"[warn] 'ground_truth' missing in {result_path}, skip iteration {iteration}")
        continue

    # correlations
    spearman_corr = df_record[['score', 'ground_truth']].corr(method='spearman').iloc[0, 1]
    pearson_corr = df_record[['score', 'ground_truth']].corr(method='pearson').iloc[0, 1]
    r_2 = r2_score(df_record['ground_truth'], df_record['score'])

    r_2_list.append(r_2)

    # OLS with intercept
    ols = LinearRegression(fit_intercept=True)
    ols.fit(df_record[['score']], df_record[['ground_truth']])
    y_pred = ols.predict(df_record[['score']]).ravel()
    r_2_intercept = r2_score(df_record['ground_truth'], y_pred)
    r_2_intercept_list.append(r_2_intercept)

    test_rmse = np.sqrt(mean_squared_error(df_record['score'], df_record['ground_truth']))

    spearman_corr_list.append(spearman_corr)
    pearson_corr_list.append(pearson_corr)
    test_rmse_list.append(test_rmse)

    # scatter plot
    ax = df_record.plot.scatter(x='score', y='ground_truth', title=f"{ccode} {target_var} {model_tag} {timeline} iter {iteration}")
    fig = ax.get_figure()
    plot_path = plot_base_dir / f"{base_name}.png"
    fig.savefig(plot_path)
    plt.close(fig)

# aggregate metrics
if len(spearman_corr_list) > 0:
    result_df_overall.loc[ccode] = (
        np.mean(spearman_corr_list), np.std(spearman_corr_list),
        np.mean(pearson_corr_list), np.std(pearson_corr_list),
        np.mean(r_2_list), np.std(r_2_list),
        np.mean(r_2_intercept_list), np.std(r_2_intercept_list),
        np.mean(test_rmse_list), np.std(test_rmse_list)
    )
    excel_path = excel_base_dir / f"{base_name}_.csv"
    result_df_overall.to_csv(excel_path, index=True, header=True)
    print(f"[info] saved overall metrics to {excel_path}")
else:
    print("[warn] No iterations with recorded metrics; nothing saved to overall metrics.")
