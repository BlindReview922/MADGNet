# MADGNet
The reproduction code of MADGNet which submitted in CVPR 2024

## Abstract
Generalizability in deep neural networks plays a pivotal role in medical image segmentation. However, deep learning-based medical image analyses tend to overlook the importance of frequency variance, which is critical element for achieving a model that is both modality-agnostic and domain-generalizable.  Additionally, various models fail to account for the potential information loss that can arise from multi-task learning under deep supervision, a factor that can impair the model’s representation ability. To address these challenges, we propose a Modality-agnostic Domain Generalizable Network (MADGNet) for medical image segmentation, which comprises two key components: a Multi-Frequency in Multi-Scale Attention (MFMSA) block and Ensemble Sub-Decoding Module (E-SDM). The MFMSA block refines the process of spatial feature extraction, particularly in capturing boundary features, by incorporating multi-frequency and multi-scale features, thereby offering informative cues for tissue outline and anatomical structures. Moreover, we propose E-SDM to mitigate information loss in multi-task learning with deep supervision, especially during substantial upsampling from low resolution. We evaluate the segmentation performance of MADGNet across six modalities and fifteen datasets. Through extensive experiments, we demonstrate that MADGNet consistently outperforms  state-of-the-art models across various modalities, showcasing superior segmentation performance.  This affirms MADGNet as a robust solution for medical image segmentation that excels in diverse imaging scenarios. 

## Overall Architecture of MADGNet
![MFMSNet](https://github.com/BlindReview922/MADGNet/assets/142275582/8c1d54c5-b03d-4c71-b7f1-81e8c91e0d36)
![CascadedDecoder](https://github.com/BlindReview922/MADGNet/assets/142275582/8c057fd3-e681-4b52-b630-591f4bc5a8f5)

## Experiment Results

### Seen Clinical Settings Results
![Screenshot from 2023-11-26 16-16-13](https://github.com/BlindReview922/MADGNet/assets/142275582/30767364-13a7-43b1-8b00-dff7aa531e7d)

### Unseen Clinical Settings Results
![Screenshot from 2023-11-26 16-15-44](https://github.com/BlindReview922/MADGNet/assets/142275582/cef29e7d-5c41-4c82-9f9a-45c45de46cb9)
