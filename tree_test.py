from datasets import load_dataset, Dataset
import pandas as pd
from trl import GRPOTrainer, GRPOConfig
import random


df = pd.read_pickle('50K_trpo_trainset.pkl')[['problem','final_answer']]
df = df[df['final_answer'].apply(lambda x: all(i.isdigit() or i=='.' for i in x))]
hf_dataset = Dataset.from_pandas(df, preserve_index=False)



def dumy_func(completions, **kwargs):
    return

training_args = GRPOConfig(output_dir="GRPO", use_vllm=True,sync_ref_model=True, num_train_epochs=1,logging_steps=20,save_steps=200, ref_model_sync_steps=600,bf16=True,num_generations=1)


trainer = GRPOTrainer(
    model="omrisap/Qwen2.5-Math-1.5B-1K-SFT",
    reward_funcs=dumy_func,
    train_dataset=hf_dataset,
    args=training_args,

)
trainer.train()

