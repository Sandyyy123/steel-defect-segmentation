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
