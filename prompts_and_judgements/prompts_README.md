# Prompts and Judgements — LLMs as Hallucination Evaluators

Materials for the study in **[Aerial Mirage: Unmasking Hallucinations in Large Vision Language Models](https://openaccess.thecvf.com/content/WACV2025/papers/Basak_Aerial_Mirage_Unmasking_Hallucinations_in_Large_Vision_Language_Models_WACV_2025_paper.pdf)** (WACV 2025) that asks how reliably advanced LLMs such as GPT-4 can rate the degree of hallucination in captions produced by another LVLM.

Because this evaluation needs dense ground truth (reference captions **and** object bounding boxes), it is run on **202 images from COCO val2017**, where such annotations exist. LLaVA generates the candidate description; the LLM is asked to score it.

## Contents

```
Prompts and judgement/
├── PROMPTS_hal_eval/     # 202 evaluation prompts, one per image  ({image_id}.txt)
├── GPT_judgements/       # GPT-3.5-Turbo and GPT-4 responses      ({image_id}_GPT.txt)
├── Gemini_judgements/    # Gemini-Pro and Gemini-Pro-Vision responses ({image_id}_Gemini.txt)
├── gpt_judge.py          # queries the OpenAI models
└── gemini_judge.py       # queries the Gemini models
```

Filenames share the same COCO `image_id` across all three folders, so a prompt and its judgements line up directly.

## Prompt design

Each prompt casts the model as a hallucination annotator and supplies:

1. **Reference captions** — the COCO ground-truth captions.
2. **Bounding boxes** — object labels with `[x, y, width, height]` coordinates.
3. **Candidate description** — the LLaVA-generated caption to be judged.

The model is asked to score the candidate on a **0–10 hallucination scale** (0 = no hallucination, 10 = maximum) and to list the specific mistakes. Judgement is grounded in object types, counts, actions, relations, and absolute/relative positions.

## Judges

| Judge | Input |
|---|---|
| GPT-3.5-Turbo | prompt only (text) |
| GPT-4 | prompt only (text) |
| Gemini-Pro | prompt only (text) |
| Gemini-Pro-Vision | prompt + the image |

For the vision judge, `gemini_judge.py` injects an extra instruction defining hallucination as any disagreement between the image and the candidate description, then passes the image alongside the prompt. Each output file holds both variants for that model, separated by a horizontal rule.

## Running

```bash
pip install openai google-generativeai pillow pandas
```

Add your API key at the top of the relevant script (`openai.api_key` / `genai.configure`), point the input and output paths at this folder, and run:

```bash
python gpt_judge.py
python gemini_judge.py
```

The scripts append to their output files, so clear or rename previous runs before re-running.

**Note:** input/output paths inside the scripts are hard-coded to the original machine and use an earlier folder name for the prompts. Update them to `PROMPTS_hal_eval/` and your own image directory before use.

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
