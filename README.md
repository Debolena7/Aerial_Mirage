The images of AeroCaps dataset can be found [here!](https://drive.google.com/drive/folders/1j7vDx2D33qE_RB-MsDds3g4yN5ZdHekK?usp=sharing) The groundtruth (reference) captions are present in AeroCaps.json file.

The LVLM-generated image-captions along with our hallucination-labelled annotations for the train and test set is released as the Labelled Illusion Dataset (LID). For LID, we use images from both the AeroCaps and the VisDrone datasets. The AeroCaps images are already available at the link above. The remaining images that were used from the VisDrone val set can be found in this [link](https://drive.google.com/file/d/1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59/view?usp=sharing).



[AeroCaps](https://huggingface.co/datasets/NLIP-lab/AeroCaps) and [LID](https://huggingface.co/datasets/NLIP-lab/LID) are also released on HuggingFace!


If you use any of these datasets in your research, please cite:

```bibtex
@InProceedings{Debolena_WACV25,
    author    = {Basak, Debolena and Bhatt, Soham and Kanduri, Sahith and Desarkar, Maunendra Sankar},
    title     = {Aerial Mirage: Unmasking Hallucinations in Large Vision Language Models},
    booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025},
    pages     = {5500-5508}
}
```
