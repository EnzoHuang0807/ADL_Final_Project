# ADL_Final_Project

[Slide](https://docs.google.com/presentation/d/1fx9ZbUw5NwrnNM_ToqwlM_iU8U4HYBtASFuQzpZ0l9U/edit?fbclid=IwAR23NHN6lBcsUm-KmlHzaC3VZ6hLrHbPaJmPLwZ8XHgaDYoR6csOXgUn7zM#slide=id.g2649c5aaa7d_3_500)

## Cosine Similarity

```
cd cos_similarity
pip install tensorflow tensorflow_hub
python classify.py
```
the result is stored at **similar_pair.json**

## GCG Experiment

```
cd GCG
```

* Download Model
```
python3 download_models.py
```

* Environment
```
pip install -e .
pip install livelossplot
```

* Data
Our orginal data is llm-attachs-main/data/similar_pair.json. By the following instruction, we can transfer the json file to 3 csv files. 
```
cd data
python3 add_newgoal.py
python3 transform_to_csv.py
```

* Train
```
cd launch_script
bash run_individual.sh vicuna behaviors path/to/your/training_data.csv path/to/your/result_folder
```

* Result
```
cd result
python3 add_result.py path/to/your/first_result_folder '../../data/similar_pair.json' 'origin_result.json'
python3 add_result.py path/to/your/second_result_folder '../../data/origin_result.json' 'similar_result.json'
python3 add_result.py path/to/your/third_result_folder '../../data/similar_result.json' 'gcg_result.json'
```
The gcg_result.json will be the final result.


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

