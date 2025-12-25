# CLIP-based Image Classification on CIFAR-10

**Date:** December, 2025  
**Author:** Sina Mohammadi  

---

## Introduction

This project implements a Python-based pipeline that utilizes the OpenCLIP implementation of CLIP (Contrastive Language-Image Pre-training) to classify CIFAR-10 images using textual descriptions. The primary model evaluated is **ViT-B/32** with the `laion2b_s34b_b79k` pre-trained weights from the LAION-2B dataset.  

The application supports **zero-shot classification** and **linear probing**, while also allowing analysis of how changes in textual prompts impact performance. The codebase is organized in a modular way for clarity and ease of future development. It is accompanied by a well-documented README that explains installation, usage, and expected results.  

---

## Implementation Templates

The project is driven by a `main.py` script that parses command-line arguments and manages different evaluation methods. The main components are:

- **Model Loading (`clip_model.py`)**: Loads the OpenCLIP model, tokenizer, and image preprocessing transformations.  
- **Data Handling (`cifar10.py`)**: Automatically downloads CIFAR-10 if needed and creates data loaders with a 45,000/5,000 train/validation split from the original training set.  
- **Embedding Computation (`embeddings`)**: Functions for extracting frozen image embeddings and computing text embeddings for class name templates and sentences.  
- **Classification Methods (`methods`)**:  
  - Zero-shot classification via cosine similarity between image and text embeddings.  
  - Linear probing using scikit-learn's `LogisticRegression` on frozen image features.  
  - Additional methods, including MLP probes on image features and learnable projection MLP heads for both text and images, for broader comparison.  

All methods keep the CLIP backbone frozen. Embeddings are **L2-normalized**. Multiple text templates can be supplied through the `--templates` argument, with embeddings averaged across templates for robustness.  

---

## Results

Experiments were conducted with **ViT-B/32 (`laion2b_s34b_b79k`)** on the CIFAR-10 test set:

- **Zero-shot:** approximately **93.6%** test accuracy. The accuracy did not improve considerably, even when changing the number or diversity of templates, which can also be verified by setting the `--template` argument in the project. I tested different configurations, including up to 300 templates. 
Shared Template Results(a photo of a {classname}.): 
```bash
  airplane   : 0.9530
  automobile : 0.9800
  bird       : 0.8990
  cat        : 0.8510
  deer       : 0.9160
  dog        : 0.9220
  frog       : 0.9290
  horse      : 0.9860
  ship       : 0.9510
  truck      : 0.9810
  ```

- **Linear probe** using Logistic Regression on frozen image embeddings: approximately **96.4%** test accuracy. This represents a substantial ~2.8% gain over the single-prompt zero-shot baseline, highlighting the value of minimal supervised adaptation.  
Shared Template Results(a photo of a {classname}.): 
```bash
  airplane   : 0.9750
  automobile : 0.9841  
  bird       : 0.9616   
  cat        : 0.9127   
  deer       : 0.9553  
  dog        : 0.9254    
  frog       : 0.9711   
  horse      : 0.9760   
  ship       : 0.9890    
  truck      : 0.9909   
  ```
- **MLP probe** on frozen image embeddings: approximately **96.8%** test accuracy. This represents a further ~0.4% gain over the Linear probe baseline, highlighting the benefit of additional flexibility in adaptation through more complex learnable structures. I also have investigated what happened if we unfreeze CLIP during training the this MLP head. but it doesn't offer a significant performance superiority. 
Shared Template Results(a photo of a {classname}.): 
```bash
  airplane  : 0.9880
  automobile: 0.9930
  bird      : 0.9480
  cat       : 0.9290
  deer      : 0.9630
  dog       : 0.9360
  frog      : 0.9790
  horse     : 0.9750
  ship      : 0.9910
  truck     : 0.9790
  ```


- **Inference based on features obtained from both images and texts:** approximately **96.7%** test accuracy. This represents a ~0.4% gain over the Linear probe baseline. However, it does not show superiority compared to the MLP probe. This baseline might be more practical for more complex datasets, where achieving high zero-shot accuracy is more challenging.  
Shared Template Results(a photo of a {classname}.):
```bash
  airplane  : 0.9810
  automobile: 0.9930
  bird      : 0.9550
  cat       : 0.9260
  deer      : 0.9690
  dog       : 0.9310
  frog      : 0.9720
  horse     : 0.9770
  ship      : 0.9900
  truck     : 0.9760
```
---
- **IExplore the variations in the way the class names could influence the model performance** : by using **per_class** mode which can be set by **template_mode** argument, you can set respected for each class differently by your self. maybe it seems, by doing it we can achieve better performance. but based on some tests which I've done by this argument, I couldn't find a general statement about superiority of a set of sentences in compare others. it's so hard to compare them. Indeed, by changing the sentences respected to each class, we will achieve different accuracies in different calsses, some of them increases but some of them decreases. achieving better overall results is challenging and unstable or at least isn't interpretable with human language. for example with this set of sentences:
```bash
"an airplane is flying.",
"a car driving on the road.",
"a photo of a bird.",
"a photo of a cat.",
"a photo of a deer.",
"a photo of a dog.",
"a photo of a frog.",
"a horse in a field.",
"a ship sailing on the sea.",
"a truck on the highway."
```

```bash
Zero-shot accuracy: 0.9237
Per-class accuracy:
  airplane  : 0.8870
  automobile: 0.9830
  bird      : 0.8990
  cat       : 0.8570
  deer      : 0.9150
  dog       : 0.9320
  frog      : 0.9320
  horse     : 0.9110
  ship      : 0.9500
  truck     : 0.9710
```
## Discussion: High Zero-Shot Performance on CIFAR-10

The observed zero-shot accuracy (~93.6%) is notably higher than what we would expected on a zero-shot inferencing. This result can be primarily attributed to the use of a modern OpenCLIP model trained on the large-scale LAION-2B dataset, which likely contains a substantial number of CIFAR-like images and semantically similar image–text pairs. Additionally, CIFAR-10 consists of a small number of concrete and visually distinctive object categories with short, unambiguous class names which leads to high performances. 

---

## Potential Improvements and Future Work
Several extensions could be explored in future work. Evaluating the same pipeline on more challenging datasets (e.g., CIFAR-100 or fine-grained classification benchmarks) would provide a more stringent test of zero-shot generalization. Additionally, analyzing robustness under distribution shifts (e.g., image corruptions) could further clarify the limits of CLIP-based zero-shot inference. 