# Caption Generation Code

Scripts used to generate the LVLM captions analysed in **[Aerial Mirage: Unmasking Hallucinations in Large Vision Language Models](https://openaccess.thecvf.com/content/WACV2025/papers/Basak_Aerial_Mirage_Unmasking_Hallucinations_in_Large_Vision_Language_Models_WACV_2025_paper.pdf)** (WACV 2025).

The captions produced here are the ones that were manually annotated for fine-grained hallucinations and released as the **Labelled Illusion Dataset (LID)** in the repository root.

## Models

| Model | Checkpoint |
|---|---|
| LLaVA-1.5 | `liuhaotian/llava-v1.5-13b` |
| InstructBLIP | `Salesforce/instructblip-vicuna-7b` |

## Contents

```
code/
├── LLaVA/
│   ├── predict.py                   # LLaVA-1.5-13B loader + Predictor class
│   ├── test_predict.py              # captions AeroCaps / VisDrone images
│   └── test_predict_obj_prompts.py  # object-conditioned prompting (VisDrone)
└── InstructBLIP/
    ├── sample_test.py               # single-image sanity check
    ├── test_aerocaps.py             # captions AeroCaps images
    └── test_visdrone.py             # captions VisDrone val images
```

`predict.py` is adapted from the official [LLaVA](https://github.com/haotian-liu/LLaVA) inference wrapper. The other scripts are ours.

## Prompts

Two caption prompts were used for both models:

- `Describe the image briefly.`
- `Describe the image in less than 30 words.`

`test_predict_obj_prompts.py` additionally builds an **object-conditioned prompt** by reading the VisDrone detection annotations, mapping category IDs to class names, and prepending the object list:

Example:
```
The objects present in the image are ['pedestrian', 'car', 'van']. Describe the image briefly.
```

This is the setting used to test whether supplying ground-truth object names reduces hallucination.

## Decoding settings

| | LLaVA-1.5-13B | InstructBLIP-Vicuna-7B |
|---|---|---|
| sampling | on | on |
| temperature | 0.2 | 0.2 |
| top-p | 1.0 | 1.0 |
| beams | – | 5 |
| repetition penalty | – | 1.5 |
| max tokens | 1024 | 1024 |

## Running

```bash
pip install torch transformers pillow
# LLaVA additionally requires the LLaVA package: https://github.com/haotian-liu/LLaVA

python InstructBLIP/test_aerocaps.py
python LLaVA/test_predict.py
```

Each script writes captions to a `.csv` and a `.json` file, keyed by image name and image ID.

Before running: Update image directories and output paths (`folder_dir`, `csv_file`, and the output JSON path) to your own locations. Set `device = "cuda"` if a GPU is available.

## Data

- AeroCaps images: [Google Drive](https://drive.google.com/drive/folders/1j7vDx2D33qE_RB-MsDds3g4yN5ZdHekK?usp=sharing) · [HuggingFace](https://huggingface.co/datasets/NLIP-lab/AeroCaps)
- VisDrone (DET val): [Google Drive](https://drive.google.com/file/d/1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59/view?usp=sharing)

## Citation

```bibtex
@InProceedings{Basak_2025_WACV,
    author    = {Basak, Debolena and Bhatt, Soham and Kanduri, Sahith and Desarkar, Maunendra Sankar},
    title     = {Aerial Mirage: Unmasking Hallucinations in Large Vision Language Models},
    booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025},
    pages     = {5500-5508}
}
```
