import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import json
import openai
from openai import AzureOpenAI, OpenAI
import pandas as pd
import numpy as np
import re
from modules import *
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from utils import query_gpt_openai, get_prompt_for_inference_unsupervised, get_prompt_for_inference_in_context, save_dataset, random_selection_nightlight,read_config, get_prompt_for_inference_in_context, random_selection_GRDP
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()


parser.add_argument('--ccode', required=True, help='ccode',default = '')
parser.add_argument('--ablation', required=False, help='ablation',default = '')
parser.add_argument('--target_var', required=True, help='testing variable')
parser.add_argument('--model', required=True, help='llm model')
parser.add_argument('--timeline', required=True, help='timeline using')


args = parser.parse_args()
if args.ablation:
    ablation = '_' + args.ablation
else:
    ablation = ''

target_var = args.target_var
ccode = args.ccode
model = args.model
timeline = args.timeline

model = model.split('/')[-1]

file_path_var = target_var
col_var = target_var
adm_offset = ''
# adm_offset = '_adm1'
iter_num = 3
iter_init_offset = 2
random_seed = range(iter_init_offset,iter_init_offset+iter_num)

# ask_num = 10ee

# prompt_version = '1'


endpoint = ""

subscription_key = ""
api_version = "2025-04-01-preview"
azure_models = ['o4-mini', 'Llama-3.3-70B-Instruct']
hf_models = ["Qwen/QwQ-32B"]
if model in azure_models: 
    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=subscription_key,
    )
else:
    client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=''
    )

prompt_version = '2'


# target_var = 'GRDP'
# target_var_desc = 'a regional GDP (in milion dollars)'
if target_var =='GRDP':
    target_var_desc = 'a regional GDP (in milion dollars)'
elif target_var =='population':
    target_var_desc = 'the population'




shot = 5
# gpt_vesrion = 'o1-preview'

# gpt_vesrion = 'gpt-4o'

fewshot_target_question = f"Infer {target_var_desc} from given location's description by utilizing multiple data aspects of given data and including any additional information you have about the region on {timeline}. Following are ground truths of other regions of {ccode}. Show your reasoning process."
# fewshot_target_question = f"Infer {target_var_desc} from given location's description by utilizing multiple data aspects of given data and including any additional information you have about the region. Following are ground truths of other regions of {ccode}. Answer the numeric score only."
# df = pd.read_csv(f"/home/donggyu/donggyu/RegionalEstimationLLM/extracted_paragraphs/NK/{ccode}_{target_var}_{ccode}_adm2_{timeline}_paragraph_output.csv", dtype={'area_id': str})
df = pd.read_csv(f"/home/donggyu/donggyu/RegionalEstimationLLM/extracted_paragraphs/20251111/{ccode}_{model}_{target_var}_{timeline}_paragraph_output.csv", dtype={'area_id': str})
# df = pd.read_csv(f"/home/donggyu/donggyu/RegionalEstimationLLM/extracted_paragraphs/{target_ccode}_GRDP_adm2_paragraph_output_9549_addr.csv")

# ground_truth = pd.read_csv('/home/donggyu/donggyu/RegionalEstimationLLM/data/label/PRK_10cities_GRDP_2020_8cities.csv', encoding='utf-8', dtype={'area_id': str})[['area_id', col_var]]
ground_truth = pd.read_csv(f'./data/label/{ccode}_{file_path_var}_{timeline}{adm_offset}.csv', encoding='utf-8', dtype={'area_id': str})[['area_id', col_var]]
ground_truth = ground_truth[ground_truth['GRDP'].notna()]
# print(ground_truth.shape)
#ground_truth[col_var] = np.log(ground_truth[col_var])
result_df_overall = pd.DataFrame(columns=['spearman_corr_mean','spearman_corr_std','pearson_corr_mean', 'pearson_corr_std', 'r2_mean','r2_std', 'r2_intercept','r2_intercept_std','test_rmse_mean','test_rmse_std'], index=[ccode])
COT_reasonings = pd.DataFrame(columns=["desc"])
for iteration in range(iter_num): 
    # print(iteration)
    if os.path.exists(f'./results/NK/test/{ccode}_{target_var}{ablation}_{model}_{shot}shots_{timeline}_iter_num_{iteration}.csv'):
        continue
    
    # random_ids = random_selection_GRDP('/home/donggyu/donggyu/RegionalEstimationLLM/data/label/PRK_GRDP_2019.csv',shot,random_seed[iteration])
    # cities_3 = ['14001','11001']
    # random_ids = random_selection_nightlight(f'./data/proxy/{ccode}_nightlight{adm_offset}.json',shot,random_seed[iteration])
    # random_ids = ['14001_20','12001_20','11001_20','14001_19','12001_19','11001_19']
    # print(random_ids)
    # target_ids = ['12001','14001','11001']
    # import random
    # random_ids = random.sample(["12001", "13009", "01008", "03004",  "09010", "04015", "04017", "10013"], 2)
    # random_ids = random_selection_nightlight(f'./data/proxy/{ccode}_nightlight{adm_offset}.json',shot,random_seed[iteration], ccode)
    # random_int_ids = [int(x) for x in random_ids]
    # print(random_ids)

    X_train = ground_truth[ground_truth['area_id'].isin(random_ids)]['area_id']
    X_test = ground_truth[ground_truth['area_id'].isin(random_ids)==False]['area_id']  
    y_train = ground_truth[ground_truth['area_id'].isin(random_ids)][col_var]
    y_test = ground_truth[ground_truth['area_id'].isin(random_ids)==False][col_var]


    X_train = X_train.reset_index().drop('index', axis=1)
    X_test = X_test.reset_index().drop('index', axis=1)
    y_train = y_train.reset_index().drop('index', axis=1)
    y_test = y_test.reset_index().drop('index', axis=1)

    df_record = pd.DataFrame(X_test.copy())
    df_record['score'] = [-1 for _ in range(len(df_record))]
    
    for i in tqdm(range(len(X_test))):
        # print(X_test)
        # print(df)
        inference_prompt = get_prompt_for_inference_in_context(df, df_record, i, X_train, y_train, X_test, fewshot_target_question)
        # inference_prompt = get_prompt_for_inference_in_context(df, df_record, i, X_train, y_train, X_test, fewshot_target_question)
        # inference_prompt += '\n I'
        # print(inference_prompt)
        # assert(1==0)
        # inference_prompt = get_prompt_for_inference_in_context(df, df_record, i, X_train, y_train, X_test, fewshot_target_question, ccode)


        model_output =  query_gpt_openai([inference_prompt], client, model=model, temperature=0.5, max_completion_tokens=25000, max_try_num=10)[0]
        model_output = model_output.replace(',','')
        # print(model_output)
        # break
        COT_reasonings.loc[i] = model_output
        COT_reasonings.to_csv(f"/home/donggyu/donggyu/RegionalEstimationLLM/COT_reasoning/NK/test/{ccode}_{target_var}_{model}_{iteration}{ablation}.csv")
        try:
            candidates = float(re.findall(r"[-+]?\d*\.\d+|\d+", model_output)[-1])
            answer = candidates
        except:
            answer = -1
        
        df_record.loc[i, 'score'] = answer
        # print(df_record)
    
        df_record['ground_truth'] = y_test.to_numpy()
        # df_record.to_csv(f'./results/new_prompt/{target_ccode}_{target_var}_paragraph_output_with_score_only_addr_{len(X_train)}shots_iter_num_{iteration}.csv', index=False, header=True)
        df_record.to_csv(f'./results/NK/test/{ccode}_{target_var}{ablation}_{model}_{len(X_train)}shots_{timeline}_iter_num_{iteration}.csv', index=False, header=True)
spearman_corr_list, pearson_corr_list, r_2_list, r_2_intercept_list,test_rmse_list = [],[],[],[],[]
for iteration in range(iter_num): 
    df_record = pd.read_csv(f'./results/NK/test/{ccode}_{target_var}{ablation}_{model}_{len(X_train)}shots_{timeline}_iter_num_{iteration}.csv')
    spearman_corr = df_record[['score', 'ground_truth']].corr(method='spearman').iloc[0][1]
    pearson_corr = df_record[['score', 'ground_truth']].corr(method='pearson').iloc[0][1]
    r_2 = r2_score(df_record['ground_truth'], df_record['score'])
    r_2_list.append(r_2)
    
    ols = LinearRegression(fit_intercept=True)  
    ols.fit( df_record[['score']], df_record[['ground_truth']])
    y_pred = ols.predict(df_record[['score']]).ravel()
    r_2_intercept = r2_score(df_record['ground_truth'], y_pred)
    r_2_intercept_list.append(r_2_intercept)

    test_rmse = np.sqrt(mean_squared_error(df_record['score'], df_record['ground_truth']))
    spearman_corr_list.append(spearman_corr)
    pearson_corr_list.append(pearson_corr)
    test_rmse_list.append(test_rmse)
    df_record.plot.scatter(x = 'score', y = 'ground_truth')
    plt.savefig(f'./results_plot/NK/test/{ccode}_{target_var}{ablation}_{model}_{len(X_train)}shots_{timeline}_{iteration}.png')
result_df_overall.loc[ccode] = np.mean(spearman_corr_list),np.std(spearman_corr_list),np.mean(pearson_corr_list),np.std(pearson_corr_list),np.mean(r_2_list), np.std(r_2_list) ,np.mean(r_2_intercept_list), np.std(r_2_intercept_list), np.mean(test_rmse_list), np.std(test_rmse_list)
result_df_overall.to_csv(f'./results_excel/NK/test/{ccode}_{target_var}{ablation}_{model}_{len(X_train)}shots{timeline}_.csv', index=True, header=True)
