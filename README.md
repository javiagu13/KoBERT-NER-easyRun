# KoBERT-NER Docker Wrapper

This repository provides a **Dockerized version of [monologg/KoBERT-NER](https://github.com/monologg/KoBERT-NER)**.  
It was created to address compatibility issues: KoBERT often requires outdated libraries, which makes setup difficult in modern environments.  
By wrapping it in Docker, the model can be run anywhere with minimal effort.

This setup has been successfully tested on AWS (containerized inside a Jupyter Notebook).

---

## 🚀 Build & Run the Container

**Build the image:**
```bash
sudo docker build -t bert_pipeline -f kobert_docker_cu101.dockerfile .
```

**Run in interactive mode (with GPU support):**
```
sudo docker run -it --gpus all bert_pipeline bash
```

---

## 📥 Model Setup

KoBERT requires downloading the model files. Inside the Docker container:

Install Git:

```
conda install git
```

Install Git LFS:

```
conda install -c conda-forge git-lfs
git lfs install
```

Clone the KoBERT repository:

```
git clone https://huggingface.co/monologg/kobert
```

The model should be placed in:

```
KoBERT-NER/kobert
```

---

## ⚙️ Modifications

The original repository crashes when auto-loading the model.
To fix this, the model is loaded locally by editing utils.py:

```
MODEL_PATH_MAP = {
    'kobert': './kobert',  # Added for local loading
    'distilkobert': 'monologg/distilkobert',
    'bert': 'bert-base-multilingual-cased',
    'kobert-lm': 'monologg/kobert-lm',
    'koelectra-base': 'monologg/koelectra-base-discriminator',
    'koelectra-small': 'monologg/koelectra-small-discriminator',
}
```

Make sure monologg/kobert is changed to ./kobert in utils.py.

---

## 🏋️ Training

Run training and evaluation:

```
python3 main.py --model_type kobert --do_train --do_eval
```

---

## 🔮 Prediction

Run inference with your own files:

```
python3 predict.py \
  --input_file {INPUT_FILE_PATH} \
  --output_file {OUTPUT_FILE_PATH} \
  --model_dir {SAVED_CKPT_PATH}

```

---

## 💻 Notes

- Training required ~14GB of GPU memory in my setup.

- Since everything runs inside Docker, the container is portable and can be deployed anywhere.

- Tested on AWS ml.g5.2xlarge instance.