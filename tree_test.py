# !rm Data_in_training/*
from datasets import load_dataset, Dataset
import pandas as pd
from trl import GRPOTrainer, GRPOConfig
import random


df = pd.read_pickle('15K_TRPO_trainset.pkl')[['problem','numerical_solution']]
# df = df.iloc[:4500]
hf_dataset = Dataset.from_pandas(df, preserve_index=False)



def dumy_func(completions, **kwargs):
    return

training_args = GRPOConfig(output_dir="GRPO", use_vllm=True,sync_ref_model=True, num_train_epochs=1,logging_steps=20,save_steps=1500, ref_model_sync_steps=5024,beta=0.01,bf16=True)


trainer = GRPOTrainer(
    model="omrisap/Qwen2.5-1.5B_30K_COT_SFT",
    reward_funcs=dumy_func,
    train_dataset=hf_dataset,
    args=training_args,

)
trainer.train()

