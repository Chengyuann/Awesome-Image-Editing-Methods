
A collection of academic articles, published methodology, and datasets on the subject of **Image Editing**.




Experiment and Data
- [Data](#data)

<br>

## Object and Attribute Manipulation
### 1. Training-Free Approaches
| Publication |    Paper Title     |    Guidance Set      | Combination                                                                                                            | Code/Project                                                 |
|:----:|-----------------------------------------------------------------------------------------------------------------------|:---------------:|:-------:|--------------------------------------------------------------|
| CVPR 2024 | [Focus on Your Instruction: Fine-grained and Multi-instruction Image Editing by Attention Modulation](https://arxiv.org/abs/2312.10113) |    instruction     | $F_{inv}^T+F_{edit}^{Attn}$|[Code](https://github.com/guoqincode/Focus-on-Your-Instruction)|
| CVPR 2024  | [Doubly Abductive Counterfactual Inference for Text-based Image Editing](https://arxiv.org/abs/2403.02981) |  text  | $F_{inv}^T+F_{edit}^{Blend}$ |[Code](https://github.com/xuesong39/DAC)|
| CVPR 2024 | [ZONE: Zero-Shot Instruction-Guided Local Editing](https://arxiv.org/abs/2312.16794) |    instruction     | $F_{inv}^T+F_{edit}^{Blend}$|[Code](https://github.com/lsl001006/ZONE)|
|WACV 2024  |  [ProxEdit: Improving Tuning-Free Real Image Editing with Proximal Guidance](https://arxiv.org/pdf/2306.05414) |  text  | $F_{inv}^F+F_{edit}^{Attn}$ |[Code](https://github.com/phymhan/prompt-to-prompt)|
| ICLR 2024 | [PnP Inversion: Boosting Diffusion-based Editing with 3 Lines of Code](https://arxiv.org/abs/2310.01506) |  text  | $F_{inv}^F+F_{edit}^{Attn}$ |[Code](https://github.com/cure-lab/PnPInversion)|
| CVPR 2024 | [An Edit Friendly DDPM Noise Space: Inversion and Manipulations](https://arxiv.org/abs/2304.06140) |  text  | $F_{inv}^F+F_{edit}^{Attn}$ |[Code](https://github.com/inbarhub/DDPM_inversion)|
|CVPR 2024 |  [Towards Understanding Cross and Self-Attention in Stable Diffusion for Text-Guided Image Editing](https://arxiv.org/abs/2403.03431) |  text  | $F_{inv}^F+F_{edit}^{Attn}$ |[Code](https://github.com/alibaba/EasyNLP/tree/master/diffusion/FreePromptEditing)|
| ICLR 2024 |  [Object-aware Inversion and Reassembly for Image Editing](https://arxiv.org/abs/2310.12149) |  text  | $F_{inv}^F+F_{edit}^{Blend}$ |[Code](https://github.com/aim-uofa/OIR)|
|ICLR 2024  |   [Noise Map Guidance: Inversion with Spatial Context for Real Image Editing](https://arxiv.org/abs/2402.04625) |  text  | $F_{inv}^F+F_{edit}^{Score}$ |[Code](https://github.com/hansam95/NMG)|
| CVPR 2024 |   [LEDITS++: Limitless Image Editing using Text-to-Image Models](https://arxiv.org/abs/2311.16711)  |  text  | $F_{inv}^F+F_{edit}^{Score}$ |[Code](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/ledits_pp)|
|ICLR 2024  |   [Noise Map Guidance: Inversion with Spatial Context for Real Image Editing](https://arxiv.org/abs/2402.04625) |  text  | $F_{inv}^F+F_{edit}^{Score}$ |[Code](https://github.com/hansam95/NMG)|
|ICLR 2024 | [Magicremover: Tuning-free Text-guided Image inpainting with Diffusion Models](https://arxiv.org/abs/2310.02848) |  text  | $F_{inv}^F+F_{edit}^{Score}$ |[Code]()|
|Arxiv 2023 |    [Region-Aware Diffusion for Zero-shot Text-driven Image Editing](https://arxiv.org/abs/2302.11797) |  text  | $F_{inv}^F+F_{edit}^{Optim}$ |[Code]()|
| ICCV 2023 |  [Delta Denoising Score](https://arxiv.org/abs/2304.07090) |  text  | $F_{inv}^F+F_{edit}^{Optim}$ |[Code](https://github.com/google/prompt-to-prompt/blob/main/DDS_zeroshot.ipynb)|
|CVPR 2024 |  [Contrastive Denoising Score for Text-guided Latent Diffusion Image Editing](https://arxiv.org/abs/2311.18608) |  text  | $F_{inv}^F+F_{edit}^{Optim}$ |[Code](https://github.com/HyelinNAM/ContrastiveDenoisingScore)|
|NeurIPS 2024 |  [Energy-Based Cross Attention for Bayesian Context Update in Text-to-Image Diffusion Models](https://arxiv.org/abs/2306.09869) |  text  | $F_{inv}^F+F_{edit}^{Optim}$ |[Code](https://github.com/EnergyAttention/Energy-Based-CrossAttention)|





### 2. Training-Based Approaches
| Publication |    Paper Title     |    Guidance Set    | Code/Project                                                 |
|:----:|-----------------------------------------------------------------------------------------------------------------------|:---------------:|--------------------------------------------------------------|
|ICLR 2024 |  [Guiding Instruction-Based Image Editing via Multimodal Large Language Models](https://arxiv.org/abs/2309.17102) |  instruction  | [Code](https://mllm-ie.github.io/)|
|CVPR 2024|     [SmartEdit: Exploring Complex Instruction-based Image Editing with Multimodal Large Language Models](https://arxiv.org/abs/2312.06739) |  instruction  | [Code](https://github.com/TencentARC/SmartEdit)|
|CVPR 2024 | [Referring Image Editing: Object-level Image Editing via Referring Expressions](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Referring_Image_Editing_Object-level_Image_Editing_via_Referring_Expressions_CVPR_2024_paper.html)  |  instruction  | [Code]()|
|Arxiv 2024 | [EditWorld: Simulating World Dynamics for Instruction-Following Image Editing](https://arxiv.org/abs/2405.14785)  |  instruction  | [Code](https://github.com/YangLing0818/EditWorld)|
<br>

## Attribute Manipulation:
### 1. Training-Free Approaches
| Publication |    Paper Title     |    Guidance Set      | Combination                                                                                                            | Code/Project                                                 |
|:----:|-----------------------------------------------------------------------------------------------------------------------|:---------------:|:-------:|--------------------------------------------------------------|
| SIGGRAPH 2024 |  [Cross-Image Attention for Zero-Shot Appearance Transfer](https://arxiv.org/abs/2311.03335)  |   image     |$F_{inv}^F+F_{edit}^{Attn}$ |[Code](https://github.com/garibida/cross-image-attention)|




<!-- 
### 2. Training-Based Approaches

<br> -->


## Spatial Transformation:
### 1. Training-Free Approaches
| Publication |    Paper Title     |    Guidance Set      | Combination                                                                                                            | Code/Project                                                 |
|:----:|-----------------------------------------------------------------------------------------------------------------------|:---------------:|:-------:|--------------------------------------------------------------|
| ICLR 2024 | [DragonDiffusion: Enabling Drag-style Manipulation on Diffusion Models](https://arxiv.org/abs/2307.02421) | image + user interface     | $F_{inv}^F+F_{edit}^{Score}$ |[Code](https://github.com/MC-E/DragonDiffusion)|
| ICLR 2024 | [DragDiffusion: Harnessing Diffusion Models for Interactive Point-based Image Editing](https://arxiv.org/abs/2306.14435) |    mask + user interface  | $F_{inv}^T+F_{inv}^F+F_{edit}^{Optim}$ |[Code](https://github.com/Yujun-Shi/DragDiffusion)|
| ICLR 2024 |  [DiffEditor: Boosting Accuracy and Flexibility on Diffusion-based Image Editing](https://arxiv.org/abs/2402.02583) |    image + user interface     | $F_{inv}^T+F_{inv}^F+F_{edit}^{Score}$  |[Code](https://github.com/MC-E/DragonDiffusion)|

<br>


<!-- 
### 2. Training-Based Approaches

 -->




## Inpainting:
### 1. Training-Free Approaches
| Publication |    Paper Title     |    Guidance Set      | Combination                                                                                                            | Code/Project                                                 |
|:----:|-----------------------------------------------------------------------------------------------------------------------|:---------------:|:-------:|--------------------------------------------------------------|
| CVPR 2024 | [Tuning-Free Image Customization with Image and Text Guidance](https://arxiv.org/abs/2403.12658) |   text + image + mask    | $F_{inv}^F+F_{edit}^{Blend}$  |[Code]()|
| TMLR 2023 |    [DreamEdit: Subject-driven Image Editing](https://arxiv.org/abs/2306.12624) |  text + image +mask | $F_{inv}^T+F_{inv}^F+F_{edit}^{Blend}$ |[Code](https://github.com/DreamEditBenchTeam/DreamEdit))|




### 2. Training-Based Approaches
| Publication |    Paper Title     |    Guidance Set    | Code/Project                                                 |
|:----:|-----------------------------------------------------------------------------------------------------------------------|:---------------:|--------------------------------------------------------------|
| CVPR 2024 | [Imagen Editor and EditBench: Advancing and Evaluating Text-Guided Image Inpainting](https://arxiv.org/pdf/2212.06909) |  text + mask  | [Code](https://imagen.research.google/editor/)|
| CVPR 2023 | [Reference-based Image Composition with Sketch via Structure-aware Diffusion Model](https://arxiv.org/abs/2304.09748) |  image + mask  | [Code]()|
|ICASSP 2024  | [Paste, Inpaint and Harmonize via Denoising: Subject-Driven Image Editing with Pre-Trained Diffusion Model](https://arxiv.org/abs/2306.07596)  |  text+ image + mask  | [Code](https://sites.google.com/view/phd-demo-page)|
|CVPR 2024 |  [AnyDoor: Zero-shot Object-level Image Customization](https://arxiv.org/abs/2307.09481)  |  image + mask  | [Code](https://github.com/ali-vilab/AnyDoor)|

<br>


## Style Change:
### 1. Training-Free Approaches
| Publication |    Paper Title     |    Guidance Set      | Combination                                                                                                            | Code/Project                                                 |
|:----:|-----------------------------------------------------------------------------------------------------------------------|:---------------:|:-------:|--------------------------------------------------------------|
| CVPR 2024  | [Style Injection in Diffusion: A Training-free Approach for Adapting Large-scale Diffusion Models for Style Transfer](https://arxiv.org/abs/2312.09008) |    image     | $F_{inv}^F+F_{edit}^{Attn}$| [Code](https://github.com/jiwoogit/StyleID)|




<!-- ### 2. Training-Based Approaches

<br> -->


## Image Translation:
### 1. Training-Free Approaches
| Publication |    Paper Title     |    Guidance Set      | Combination                                                                                                            | Code/Project                                                 |
|:----:|-----------------------------------------------------------------------------------------------------------------------|:---------------:|:-------:|--------------------------------------------------------------|
| CVPR 2024  |  [FreeControl: Training-Free Spatial Control of Any Text-to-Image Diffusion Model with Any Condition](https://arxiv.org/abs/2312.07536)  |    text     | $F_{inv}^F+F_{edit}^{Score}$  | [Code](https://github.com/genforce/freecontrol)|


### 2. Training-Based Approaches
| Publication |    Paper Title     |    Guidance Set    | Code/Project                                                 |
|:----:|-----------------------------------------------------------------------------------------------------------------------|:---------------:|--------------------------------------------------------------|
| AAAI 2024  |  [T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.08453) |  text  | [Code](https://github.com/TencentARC/T2I-Adapter)|
| CVPR 2024 | [SCEdit: Efficient and Controllable Image Diffusion Generation via Skip Connection Editing](https://arxiv.org/abs/2312.11392) |  text  | [Code](https://github.com/ali-vilab/SCEdit)|
|  Arxiv 2024 | [One-Step Image Translation with Text-to-Image Models](https://arxiv.org/abs/2403.12036) |  text  | [Code](https://github.com/GaParmar/img2img-turbo)|

<br>

## Subject-Driven Customization:
### 1. Training-Free Approaches
| Publication |    Paper Title     |    Guidance Set      | Combination                                                                                                            | Code/Project                                                 |
|:----:|-----------------------------------------------------------------------------------------------------------------------|:---------------:|:-------:|--------------------------------------------------------------|
| AAAI 2024 |   [Decoupled Textual Embeddings for Customized Image Generation](https://arxiv.org/abs/2312.11826) |    text     | $F_{inv}^T+F_{edit}^{Norm}$  | [Code](https://github.com/PrototypeNx/DETEX)|
| ICLR 2024 | [A Data Perspective on Enhanced Identity Preservation for Diffusion Personalization](https://arxiv.org/abs/2311.04315) |    text     | $F_{inv}^T+F_{edit}^{Norm}$  | [Code]()|
| CVPR 2024 |  [FaceChain-SuDe: Building Derived Class to Inherit Category Attributes for One-shot Subject-Driven Generation](https://arxiv.org/abs/2403.06775) |    text     | $F_{inv}^T+F_{edit}^{Norm}$  | [Code](https://github.com/modelscope/facechain)|
| Arxiv 2023 |  [ViCo: Detail-Preserving Visual Condition for Personalized Text-to-Image Generation](https://arxiv.org/abs/2306.00971) |    text     | $F_{inv}^T+F_{edit}^{Attn}$  | [Code](https://github.com/haoosz/ViCo)|
|CVPR 2024 | [DreamMatcher: Appearance Matching Self-Attention for Semantically-Consistent Text-to-Image Personalization](https://arxiv.org/abs/2402.09812) |    text     | $F_{inv}^T+F_{edit}^{Attn}$  | [Code](https://ku-cvlab.github.io/DreamMatcher/)|
|Arxiv 2024 | [Direct Consistency Optimization for Compositional Text-to-Image Personalization](https://arxiv.org/abs/2402.12004) |    text     | $F_{inv}^T+F_{edit}^{Score}$  | [Code](https://github.com/kyungmnlee/dco)|
| Arxiv 2024 |  [Pick-and-Draw: Training-free Semantic Guidance for Text-to-Image Personalization](https://arxiv.org/abs/2401.16762) |    text     | $F_{inv}^F+F_{edit}^{Optim}$   | [Code]()|



  <!-- [Cones2]() | [📖 ] | [Inversion+Editing] | [🌐 Code]()  -->


   
  <!-- [CatVersion]() | [📖 ] | [Inversion+Editing] | [🌐 Code]()  -->

  

### 2. Training-Based Approaches
| Publication |    Paper Title     |    Guidance Set    | Code/Project                                                 |
|:----:|-----------------------------------------------------------------------------------------------------------------------|:---------------:|--------------------------------------------------------------|
|ICLR 2024  |  [Taming Encoder for Zero Fine-tuning Image Customization with Text-to-Image Diffusion Models](https://arxiv.org/abs/2304.02642) |  text  | [Code]()|
| CVPR 2024 | [InstantBooth: Personalized Text-to-Image Generation without Test-Time Finetuning](https://arxiv.org/abs/2304.03411) |  text  | [Code](https://jshi31.github.io/InstantBooth/)|
|ICLR 2024  | [Enhancing Detail Preservation for Customized Text-to-Image Generation: A Regularization-Free Approach](https://github.com/drboog/ProFusion)  |  text  | [Code]()|



<br>


## Attribute-Driven Customization:
### 1. Training-Free Approaches
| Publication |    Paper Title     |    Guidance Set      | Combination                                                                                                            | Code/Project                                                 |
|:----:|-----------------------------------------------------------------------------------------------------------------------|:---------------:|:-------:|--------------------------------------------------------------|
| NeurIPS 2023 | [StyleDrop: Text-to-Image Generation in Any Style](https://arxiv.org/abs/2306.00983) |    text     | $F_{inv}^T+F_{edit}^{Norm}$  | [Code](https://styledrop.github.io/)|
 


### 2. Training-Based Approaches
| Publication |    Paper Title     |    Guidance Set    | Code/Project                                                 |
|:----:|-----------------------------------------------------------------------------------------------------------------------|:---------------:|--------------------------------------------------------------|
|ICLR 2024  | [Language-Informed Visual Concept Learning](https://arxiv.org/abs/2312.03587)  |  text  | [Code](https://cs.stanford.edu/~yzzhang/projects/concept-axes/)|
|Arxiv 2024  | [pOps: Photo-Inspired Diffusion Operators](https://arxiv.org/abs/2406.01300) |  text  | [Code](https://github.com/pOpsPaper/pOps)|


## Contact

```
issue
```
