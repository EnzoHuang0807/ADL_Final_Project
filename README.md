# ADL_Final_Project

## Cosine Similarity

```
cd cos_similarity
pip install tensorflow tensorflow_hub
python classify.py
```
the result is stored at **similar_pair.json**

## AutoDAN and Evaluation

* This directory is modified from the original [AutoDAN](https://github.com/SheltonLiu-N/AutoDAN/tree/main) repository

### Quick start

* Build environment

```
cd AutoDAN
conda create -n AutoDAN python=3.9
conda activate AutoDAN
pip install -r requirements.txt
```

* Download LLM

```
python download_models.py
```

* Format of input file
```
[
    {
        id : 0 ~ 49,
        goal : "instruction",
        target : "Sure, here is ...",
        similar_goal : "instruction",
        similar_target : "Sure, here is ...",
        prefix : (if any),
        similar_prefix : (if any),
        known_prefix : (if any),
        suffix : (if any),
        similar_suffix : (if any),
        known_suffix: (if any)
    },
]
```

* Run AutoDAN

```
python -W ignore AutoDAN.py \
  --API_key <your openai API key> \
  --low_memory 1 \
  --model <your model> \
  --input_file <your input file> \
  --output_file <your output file> \
  --log_file <your log file>
```

* Open `AutoDAN.py` to view for more arguments

* Run evaluation (ASR)

```
python -W ignore eval.py \
  --API_key  <your openai API key> \
  --low_memory 1 \
  --model <your model> \
  --input_file <your input file> \
  <other arguments> \
```

* Open `eval.py` to view for more arguments

