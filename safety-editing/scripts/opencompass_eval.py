from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.triviaqa.triviaqa_gen import \
        triviaqa_datasets # knowledge
    from opencompass.configs.datasets.Xsum.Xsum_gen import \
        Xsum_datasets # understanding
    from opencompass.configs.datasets.lambada.lambada_gen import \
        lambada_datasets # understanding
    # from opencompass.configs.datasets.obqa.obqa_gen import \
    #     obqa_datasets # 15m
    from opencompass.configs.datasets.storycloze.storycloze_gen import \
        storycloze_datasets # 10m reasoning
    from opencompass.configs.datasets.race.race_gen import \
        race_datasets # 21m exam
    from opencompass.configs.datasets.mbpp.mbpp_gen import \
        mbpp_datasets # 16m code
    from opencompass.configs.datasets.gpqa.gpqa_gen import \
        gpqa_datasets # 18m knowledge
    from opencompass.configs.datasets.humaneval.humaneval_gen import \
        humaneval_datasets # 7m code
    from opencompass.configs.datasets.winogrande.winogrande_gen import \
        winogrande_datasets # 10m language
    from opencompass.configs.datasets.strategyqa.strategyqa_gen import \
        strategyqa_datasets # 30m reasoning
    from opencompass.configs.models.qwen2_5.hf_qwen2_5_1_5b_instruct import \
        models as hf_qwen2_5_1_5b_instruct_models
    from opencompass.configs.models.qwen2_5.hf_qwen2_5_7b_instruct import \
        models as hf_qwen2_5_7b_instruct_models
    from opencompass.configs.models.hf_llama.hf_llama3_2_3b_instruct import \
        models as hf_llama3_2_3b_instruct_models
    from opencompass.configs.models.qwen3.hf_qwen3_14b_instruct import \
        models as hf_qwen3_14b_instruct_models



# USER: 填入待评测模型的本地权重目录或 Hugging Face 缓存路径
hf_qwen3_14b_instruct_models[0]["path"] = ""

# datasets = [triviaqa_datasets, Xsum_datasets, lambada_datasets, race_datasets, gpqa_datasets, humaneval_datasets]
datasets = lambada_datasets
models = hf_qwen3_14b_instruct_models

# USER: OpenCompass 运行输出目录
work_dir = ""