# References - Severstal Steel Defect Detection

Each reference verified live via CrossRef (https://api.crossref.org/works/{doi}) or Semantic Scholar Graph API (`paper/arXiv:{id}`). DOIs and arXiv IDs below resolve at the time of writing (May 2026). Volume, issue, and page numbers are intentionally omitted per project rule (keep only Author / Title / Journal / Year / DOI or PMID).

## Semantic segmentation foundations

1. **Long, J., Shelhamer, E., & Darrell, T.** (2015). Fully convolutional networks for semantic segmentation. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*. DOI: [10.1109/CVPR.2015.7298965](https://doi.org/10.1109/CVPR.2015.7298965). The foundational paper that recast segmentation as dense pixel-wise classification with fully-convolutional encoder-decoder networks.

2. **Ronneberger, O., Fischer, P., & Brox, T.** (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *Lecture Notes in Computer Science (MICCAI)*. DOI: [10.1007/978-3-319-24574-4_28](https://doi.org/10.1007/978-3-319-24574-4_28). Symmetric encoder-decoder with skip connections; the architecture used as the baseline backbone in this project and the dominant choice for industrial defect segmentation.

3. **Zhao, H., Shi, J., Qi, X., Wang, X., & Jia, J.** (2017). Pyramid Scene Parsing Network. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*. DOI: [10.1109/CVPR.2017.660](https://doi.org/10.1109/CVPR.2017.660). PSPNet introduced multi-scale context aggregation through pyramid pooling, a key idea for capturing both fine and coarse defect structures.

4. **Chen, L.-C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H.** (2018). Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation (DeepLabV3+). *Lecture Notes in Computer Science (ECCV)*. DOI: [10.1007/978-3-030-01234-2_49](https://doi.org/10.1007/978-3-030-01234-2_49). DeepLabV3+ is the alternate advanced-model backbone listed in this project; combines atrous spatial pyramid pooling with a decoder for sharper boundaries.

5. **Chen, L.-C., Papandreou, G., Schroff, F., & Adam, H.** (2017). Rethinking Atrous Convolution for Semantic Image Segmentation. arXiv:[1706.05587](https://arxiv.org/abs/1706.05587). DeepLabV3, the immediate predecessor of V3+, introducing the atrous spatial pyramid pooling block used in this project's advanced model.

6. **He, K., Gkioxari, G., Dollar, P., & Girshick, R.** (2017). Mask R-CNN. *IEEE International Conference on Computer Vision (ICCV)*. DOI: [10.1109/ICCV.2017.322](https://doi.org/10.1109/ICCV.2017.322). Instance-level mask prediction; relevant as the alternate detection-plus-segmentation framing not chosen here, since defect classes are non-instance and dense per-pixel.

7. **Wang, J., Sun, K., Cheng, T., Jiang, B., Deng, C., Zhao, Y., Liu, D., Mu, Y., Tan, M., Wang, X., Liu, W., & Xiao, B.** (2021). Deep High-Resolution Representation Learning for Visual Recognition (HRNet). *IEEE Transactions on Pattern Analysis and Machine Intelligence*. DOI: [10.1109/TPAMI.2020.2983686](https://doi.org/10.1109/TPAMI.2020.2983686). HRNet maintains high-resolution feature maps throughout the network; useful contrast against the encoder-decoder family adopted in this project.

8. **Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., & Luo, P.** (2021). SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers. arXiv:[2105.15203](https://arxiv.org/abs/2105.15203). Transformer-based hierarchical encoder with lightweight all-MLP decoder; the advanced-model backbone (`mit-b3` variant) used in this project.

9. **Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., et al.** (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. arXiv:[2010.11929](https://arxiv.org/abs/2010.11929). The Vision Transformer (ViT) paper that motivates SegFormer's transformer-encoder architecture.

10. **Oktay, O., Schlemper, J., Folgoc, L. L., et al.** (2018). Attention U-Net: Learning Where to Look for the Pancreas. arXiv:[1804.03999](https://arxiv.org/abs/1804.03999). Adds attention gates inside U-Net skip connections; the technique is directly applicable to suppressing background noise in steel-surface masks.

11. **Milletari, F., Navab, N., & Ahmadi, S.-A.** (2016). V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation. arXiv:[1606.04797](https://arxiv.org/abs/1606.04797). Introduced the soft Dice loss that the present project uses (with BCE) for the baseline.

12. **Isensee, F., Jaeger, P. F., Kohl, S. A. A., Petersen, J., & Maier-Hein, K. H.** (2020). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods*. DOI: [10.1038/s41592-020-01008-z](https://doi.org/10.1038/s41592-020-01008-z). The self-configuring U-Net pipeline; a strong industrial-CV baseline whose preprocessing and augmentation rules transfer well to non-medical segmentation.

## Backbones, building blocks, and training tricks

13. **He, K., Zhang, X., Ren, S., & Sun, J.** (2016). Deep Residual Learning for Image Recognition (ResNet). *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*. DOI: [10.1109/CVPR.2016.90](https://doi.org/10.1109/CVPR.2016.90). ResNet-34 is the encoder backbone of this project's baseline U-Net; the residual-block formulation is the de-facto standard for ImageNet-pretrained encoders.

14. **Tan, M., & Le, Q. V.** (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. arXiv:[1905.11946](https://arxiv.org/abs/1905.11946). EfficientNet-B4 is the alternate advanced-model encoder when the SegFormer pipeline is unavailable; the compound-scaling rule informs latency-vs-accuracy trade-offs.

15. **Huang, G., Liu, Z., van der Maaten, L., & Weinberger, K. Q.** (2017). Densely Connected Convolutional Networks (DenseNet). *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*. DOI: [10.1109/CVPR.2017.243](https://doi.org/10.1109/CVPR.2017.243). DenseNet is a competitive alternative encoder family; included for completeness of the backbone-comparison literature.

16. **Hu, J., Shen, L., & Sun, G.** (2018). Squeeze-and-Excitation Networks. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*. DOI: [10.1109/CVPR.2018.00745](https://doi.org/10.1109/CVPR.2018.00745). Channel-attention block widely added to defect-detection backbones for a small but consistent accuracy gain.

17. **Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C.** (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*. DOI: [10.1109/CVPR.2018.00474](https://doi.org/10.1109/CVPR.2018.00474). Lightweight encoder relevant for on-edge mill-inspection deployments where the SegFormer mit-b3 model is too heavy.

18. **Loshchilov, I., & Hutter, F.** (2017). SGDR: Stochastic Gradient Descent with Warm Restarts. arXiv:[1608.03983](https://arxiv.org/abs/1608.03983). The cosine-with-restarts schedule used by the advanced model.

## Loss functions for class imbalance

19. **Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollar, P.** (2017). Focal Loss for Dense Object Detection. *IEEE International Conference on Computer Vision (ICCV)*. DOI: [10.1109/ICCV.2017.324](https://doi.org/10.1109/ICCV.2017.324). The focal loss formulation that down-weights easy negatives, the building block for the focal Tversky loss used in this project.

20. **Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollar, P.** (2020). Focal Loss for Dense Object Detection (TPAMI extended version). *IEEE Transactions on Pattern Analysis and Machine Intelligence*. DOI: [10.1109/TPAMI.2018.2858826](https://doi.org/10.1109/TPAMI.2018.2858826). Extended focal-loss treatment with additional analysis relevant to the heavy-tailed class imbalance in industrial defect data.

21. **Sudre, C. H., Li, W., Vercauteren, T., Ourselin, S., & Cardoso, M. J.** (2017). Generalised Dice Overlap as a Deep Learning Loss Function for Highly Unbalanced Segmentations. *Lecture Notes in Computer Science*. DOI: [10.1007/978-3-319-67558-9_28](https://doi.org/10.1007/978-3-319-67558-9_28). Generalised Dice for class-imbalanced multi-label segmentation; theoretical basis for the Tversky-family losses adopted here.

22. **Hosseini, S. M., & Baghshah, M. S.** (2026). Dilated Balanced cross entropy loss for medical image segmentation. *BMC Medical Imaging*. DOI: [10.1186/s12880-026-02245-y](https://doi.org/10.1186/s12880-026-02245-y). Recent comparative loss-function study against the imbalance regimes typical of pixel-level industrial defect data.

## Pretraining, datasets, and interpretability

23. **Russakovsky, O., Deng, J., Su, H., et al.** (2015). ImageNet Large Scale Visual Recognition Challenge. *International Journal of Computer Vision*. DOI: [10.1007/s11263-015-0816-y](https://doi.org/10.1007/s11263-015-0816-y). The pretraining corpus used for both encoders in this project (ResNet-34 baseline, mit-b3 advanced).

24. **Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L.** (2009). ImageNet: A large-scale hierarchical image database. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*. DOI: [10.1109/CVPR.2009.5206848](https://doi.org/10.1109/CVPR.2009.5206848). Original ImageNet dataset paper.

25. **Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D.** (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization. *IEEE International Conference on Computer Vision (ICCV)*. DOI: [10.1109/ICCV.2017.74](https://doi.org/10.1109/ICCV.2017.74). The standard interpretability tool for convolutional defect-detection models; useful for QC-line root-cause attribution.

## Steel surface defect detection (domain literature)

26. **He, Y., Song, K., Meng, Q., & Yan, Y.** (2020). An End-to-End Steel Surface Defect Detection Approach via Fusing Multiple Hierarchical Features. *IEEE Transactions on Instrumentation and Measurement*. DOI: [10.1109/TIM.2019.2915404](https://doi.org/10.1109/TIM.2019.2915404). Hierarchical feature-fusion approach for hot-rolled steel surface defects; one of the most-cited domain papers for steel CV inspection.

27. **Lv, X., Duan, F., Jiang, J., Fu, X., & Gan, L.** (2020). Deep Metallic Surface Defect Detection: The New Benchmark and Detection Network. *Sensors*. DOI: [10.3390/s20061562](https://doi.org/10.3390/s20061562). Benchmark dataset and detection network for metallic surface defects; methodologically adjacent to the Severstal task.

28. **Iqbal, R., Maniak, T., Doctor, F., & Karyotis, C.** (2019). Fault Detection and Isolation in Industrial Processes Using Deep Learning Approaches. *IEEE Transactions on Industrial Informatics*. DOI: [10.1109/TII.2019.2902274](https://doi.org/10.1109/TII.2019.2902274). Broader industrial-process fault-detection survey covering deep-learning approaches relevant to deployment in DACH manufacturing.

29. **Qin, M., Li, H., Huang, Y., Tong, X., & Liang, Z.** (2026). Defect-Mask2Former: An Improved Semantic Segmentation Model for Precise Small-Sized Defect Detection. *Sensors*. DOI: [10.3390/s26072254](https://doi.org/10.3390/s26072254). Recent transformer-based defect segmentation architecture; direct comparator to the SegFormer choice in this project.

30. **Feng, Y., Jin, L., Yang, H., & Liu, S.** (2025). An improved steel defect detection model using multi-scale feature fusion based on YOLO-MF. *Scientific Reports*. DOI: [10.1038/s41598-025-29596-w](https://doi.org/10.1038/s41598-025-29596-w). Recent steel-specific defect-detection benchmark; bounding-box framing complements the per-pixel framing here.

31. **Li, X., Zhao, Y., Jiao, X., Meng, Q., Guo, Z., Yao, et al.** (2025). PEYOLO: A Perception-Efficient Network for Multiscale Surface Defects Detection. *Scientific Reports*. DOI: [10.1038/s41598-025-05574-0](https://doi.org/10.1038/s41598-025-05574-0). Perception-efficient detector for multiscale surface defects; reports class-imbalance handling strategies that translate to the Severstal scratches-vs-pitting imbalance.

32. **Ge, A., Lv, Y., & Huang, J.** (2026). WaveMamba-YOLO: Combining frequency awareness and state-space modeling for defect localisation. *PLOS ONE*. DOI: [10.1371/journal.pone.0344940](https://doi.org/10.1371/journal.pone.0344940). State-space and wavelet-frequency hybrid for industrial defect localisation; alternative architectural family worth tracking.

33. **Zhang, T., Wang, D., & Zhang, S.** (2026). Multi-attention cross-scanning VM-UNet for X-ray welding defect detection of steel pipelines. *PLOS ONE*. DOI: [10.1371/journal.pone.0341805](https://doi.org/10.1371/journal.pone.0341805). VM-UNet variant on welding defects; closely adjacent imaging modality (X-ray vs visible) showing transferable architectural ideas.

34. **Xie, J., Wu, P., Chen, J., Pan, Y., Zhong, H., Deng et al.** (2025). A real-time segmentation network for lithium battery surface defect detection. *Scientific Reports*. DOI: [10.1038/s41598-025-18315-0](https://doi.org/10.1038/s41598-025-18315-0). Real-time per-pixel defect segmentation in a different industrial substrate; latency engineering directly applicable to mill-line deployment.


---

## 2024-2026 additions (post-QA literature scout)

# Additional References - Severstal Steel Defect Detection (Literature Scout, Role C)

Generated independently of `reports/references.md`. Every entry below was resolved live via the CrossRef API (`https://api.crossref.org/works/{doi}`) on 2026-05-08; any candidate that did not resolve was dropped, not padded.

Format: `Authors. Title. Journal. Year. DOI:10.xxx/yyy` (no volume, issue, or pages).

## State-of-the-art callout (gaps not covered by `reports/references.md`)

The current `references.md` is strong on segmentation foundations (FCN, U-Net, DeepLabV3+, SegFormer, HRNet) and pre-2024 steel domain papers, but the following SOTA threads are missing and the project should cite them:

1. **DINOv2-based self-supervised learning for surface-defect detection** - Pieressa et al. 2026 (DOI:10.1007/s00170-026-17386-1) and Docherty et al. 2026 (DOI:10.1002/aisy.202501094) directly demonstrate self-supervised foundation features beating ImageNet-pretrained CNNs on small-data industrial segmentation. The current advanced model uses ImageNet `mit-b3`; a DINOv2-pretrained encoder is the relevant 2026 comparator.
2. **Segment Anything Model (SAM) for industrial defect segmentation** - Naddaf-Sh et al. 2025 (DOI:10.3390/s25010277) and Huang et al. 2025 (DOI:10.1007/s10845-025-02658-6) show SAM and SAM-adapter pipelines as zero/few-shot baselines. The brief lists no foundation-model comparator and the manuscript should at least benchmark against SAM-adapter.
3. **PatchCore / reconstruction-based anomaly detection as a class-agnostic pre-screen** - Kumari and Prabha 2025 (DOI:10.47852/bonviewaia52026321) and Kohler et al. 2025 (DOI:10.1016/j.jmsy.2024.12.005) cover the unsupervised-anomaly route, which is the standard industry alternative when defect labels are scarce; missing entirely from the current references.
4. **Domain-shift / drift handling for cross-mill deployment** - Barberena et al. 2025 (DOI:10.5220/0013170900003912) addresses MMD-based domain adaptation, the exact issue raised in `brief.md` Open question 4 (cross-mill transfer). No drift-handling reference is currently cited.
5. **Lightweight student-teacher distillation for edge mill-line latency** - Loganathan et al. 2025 (DOI:10.1109/ic3it66137.2025.11341671) on BiSeNetV2 + knowledge distillation maps directly onto Open question 3 (100 ms QC-line budget); the current references list lacks any 2024+ distillation paper.

These five entries are the highest-leverage additions and should anchor a "comparison with recent literature" subsection in the manuscript Discussion.

## Architectures and segmentation methods (2024-2026)

1. Zichen Dang, Xingshuo Wang. FD-YOLO11: A Feature-Enhanced Deep Learning Model for Steel Surface Defect Detection. IEEE Access. 2025. DOI:10.1109/access.2025.3559733
2. Sara Ashrafi, Sobhan Teymouri, Sepideh Etaati, Javad Khoramdel, Yasamin Borhani, Esmaeil Najafi. Steel surface defect detection and segmentation using deep neural networks. Results in Engineering. 2025. DOI:10.1016/j.rineng.2025.103972
3. Qiqi Zhou, Haichao Wang. CABF-YOLO: a precise and efficient deep learning method for defect detection on strip steel surface. Pattern Analysis and Applications. 2024. DOI:10.1007/s10044-024-01252-5
4. Shouluan Wu, Hui Yang, Liefa Liao, Chao Song, Yating Fang, Jianglong Fu. DSAT: a dynamic sparse attention transformer for steel surface defect detection with hierarchical feature fusion. Scientific Reports. 2025. DOI:10.1038/s41598-025-14935-8
5. Ning Zhang, Ziyang Liu, Enxu Zhang, Yuanqi Chen, Jie Yue. An ESG-ConvNeXt network for steel surface defect classification based on hybrid attention mechanism. Scientific Reports. 2025. DOI:10.1038/s41598-025-88958-6
6. Yan Jiang, Jiaxin Dai, Zhuoru Jiang. FAX-Net: An Enhanced ConvNeXt Model with Symmetric Attention and Transformer-FPN for Steel Defect Classification. Symmetry. 2025. DOI:10.3390/sym17081313
7. Shenglong Hou, Hua He, Kang Peng, Sibo Qiao. Improved Swin Transformer-Based Model for Hot-Rolled Strip Defect Detecting. Computing and Informatics. 2024. DOI:10.31577/cai_2024_6_1352
8. Liang Gong, Hang Dong, Xinyu Zhang, Xin Cheng, Fan Ye, Liangchao Guo. Spiking ViT: spiking neural networks with transformer-attention for steel surface defect classification. Journal of Electronic Imaging. 2024. DOI:10.1117/1.jei.33.3.033001

## Hot-rolled / strip steel domain (2024-2026)

9. Xiaoyan Zhu, Xin Wan, Mingyu Zhang. EMC-YOLO: a feature enhancement and fusion based surface defect detection for hot rolled strip steel. Engineering Research Express. 2025. DOI:10.1088/2631-8695/ada7c4
10. Wenzheng Sun, Na Meng, Longfa Chen, Sen Yang, Yuguo Li, Shuo Tian. CTL-YOLO: A Surface Defect Detection Algorithm for Lightweight Hot-Rolled Strip Steel Under Complex Backgrounds. Machines. 2025. DOI:10.3390/machines13040301
11. Huanwei Xu, Xuyuan Xiao, Zewei Zhao, Zhonglai Wang. YOLOv8n-GAM: an improved surface defect detection network for hot-rolled strip steel. Engineering Research Express. 2024. DOI:10.1088/2631-8695/ad5417

## Foundation models for industrial defect (SAM, DINOv2)

12. Amir-M. Naddaf-Sh, Vinay S. Baburao, Hassan Zargarzadeh. Leveraging Segment Anything Model (SAM) for Weld Defect Detection in Industrial Ultrasonic B-Scan Images. Sensors. 2025. DOI:10.3390/s25010277
13. Fan Huang, Liming Zheng, Haiying Wen, Min Dai, Zhisheng Zhang. A novel data augmentation method for few-shot industrial surface defect detection based on segment anything model adapter. Journal of Intelligent Manufacturing. 2025. DOI:10.1007/s10845-025-02658-6
14. Andrea Pieressa, Chung-Yin Lin, Giovanni Lucchetta, Lih-Sheng Turng. A DINOv2-based self-supervised learning framework for automated detection of surface defects in injection molding. The International Journal of Advanced Manufacturing Technology. 2026. DOI:10.1007/s00170-026-17386-1
15. Ronan Docherty, Antonis Vamvakeros, Samuel J. Cooper. Upsampling DINOv2 Features for Unsupervised Vision Tasks and Weakly Supervised Materials Segmentation. Advanced Intelligent Systems. 2026. DOI:10.1002/aisy.202501094

## Anomaly detection (unsupervised baselines)

16. Markus Kohler, Dionysios Mitsios, Christian Endisch. Reconstruction-based visual anomaly detection in wound rotor synchronous machine production using convolutional autoencoders and structural similarity. Journal of Manufacturing Systems. 2025. DOI:10.1016/j.jmsy.2024.12.005
17. Shalini Kumari, Chander Prabha. Anomaly Detection Utilizing PatchCore for Reimagining Industrial Visual Inspection. Artificial Intelligence and Applications. 2025. DOI:10.47852/bonviewaia52026321

## Domain adaptation and drift handling

18. Xuban Barberena, Fatima Saiz, Inigo Barandiaran. Handling Drift in Industrial Defect Detection Through MMD-Based Domain Adaptation. Proceedings of the 20th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications. 2025. DOI:10.5220/0013170900003912

## Class-imbalance loss functions (2024-2025)

19. Feilong Xu, Feiyang Yang, Xiongfei Li, Xiaoli Zhang. A Unified Loss for Handling Inter-Class and Intra-Class Imbalance in Medical Image Segmentation. Proceedings of the AAAI Conference on Artificial Intelligence. 2025. DOI:10.1609/aaai.v39i8.32956
20. Mariano Cabezas, Yago Diez. An Analysis of Loss Functions for Heavily Imbalanced Lesion Segmentation. Sensors. 2024. DOI:10.3390/s24061981

## Edge deployment, distillation, and ensembles

21. A. S. Loganathan, Mohd Athar Nawab Jan, Vasavi Cheruku, K. Ramya Laxmi, K. Deiwakumari. Enhanced Lightweight Semantic Segmentation Using BiSeNetV2 with Knowledge Distillation for Real-Time Edge Deployment. 2025 International Conference on Communication, Computer, and Information Technology (IC3IT). 2025. DOI:10.1109/ic3it66137.2025.11341671
22. Tajmal Hussain, Jongwon Seok. Steel Surface Defect Recognition in Smart Manufacturing Using Deep Ensemble Transfer Learning-Based Techniques. Computer Modeling in Engineering & Sciences. 2025. DOI:10.32604/cmes.2024.056621
23. Hui Zuo, Qipei Mei, Nima Shirzad-Ghaleroudkhani. Automated Structural Inspection with Mask2Former and VGGT for Drone-Assisted Surface Defect Segmentation and 3D Visualization. AMPP Alberta Conference. 2026. DOI:10.5006/ac2026_00071

## Surveys and systematic reviews (2024-2025)

24. Emine Asar, Atilla Ozgur. Systematic Review of Steel Surface Defect Detection Methods on the Open Access Datasets of Severstal and the Northeastern University (NEU). Engineering Materials. 2024. DOI:10.1007/978-3-031-57468-9_3
25. Feiyu Chen, Lei Fu, Yingqian Zhang, Jiaqi Li, Qian Zhang, Shihao Bi. A Review of Deep Learning-Based Steel Surface Defect Detection. Academic Journal of Science and Technology. 2025. DOI:10.54097/g36nm962

