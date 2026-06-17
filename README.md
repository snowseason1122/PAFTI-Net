# PAFTI-Net: A Parallel Deep Learning Model for Termite Infestation Level Prediction Using Gas Monitoring Data
> Official Project Repository |  Paper

<div align="center">
  <!-- <img src="https://img.shields.io/badge/Journal-IEEE%20Access-blue" alt="Journal"> -->
  <!-- <img src="https://img.shields.io/badge/DOI-10.1109%2FACCESS.2024.0429000-orange" alt="DOI"> -->
  <img src="https://img.shields.io/badge/Python-3.9-green" alt="Python">
  <img src="https://img.shields.io/badge/Framework-PyTorch%202.0-red" alt="PyTorch">
  <img src="https://img.shields.io/badge/Task-Time%20Series%20Classification-purple" alt="Task">
</div>

---

## 📖 Abstract / 摘要
Traditional termite monitoring mainly relies on qualitative alarm, which fails to quantify infestation levels. Gas monitoring time-series data is vulnerable to noise interference, and existing methods struggle to jointly model local and global temporal patterns. To address these issues, this work constructs a gas monitoring platform to collect $CO_2$, $CH_4$, temperature and humidity time-series data, and proposes **PAFTI-Net (Parallel Adaptive Fusion and Temporal Interaction Network)**.

The model adopts an adaptive feature fusion module to suppress noise and redundant information, and designs three parallel branches to capture local variations, long-range global dependencies and multi-factor nonlinear interactions. Finally, dynamic fusion is applied to output three termite infestation risk levels (low, medium, high). Experimental results show that PAFTI-Net achieves **92.31% accuracy** and **89.71% F1-score** on the test set, outperforming mainstream baseline models. This work realizes the transformation from qualitative alarm to graded early warning, providing effective technical support for intelligent termite prevention and control.

---

## 👥 Authors & Affiliations / 作者与单位
**Authors**: Shan Wu, Hangjun Wang, Weiling Lu
**Corresponding Author**: Hangjun Wang (whj@zafu.edu.cn)

1. College of Mathematics and Computer Science, Zhejiang A&F University, Hangzhou 311300, China
2. School of Electronic Information Engineering, Huzhou College, Huzhou 313000, China
3. Ji Yang College, Zhejiang A&F University, Shaoxing 311800, China

### 💰 Funding / 基金支持
This research was supported by Zhejiang Provincial Natural Science Foundation of China under Grant No. LY23C140004.

---

## 🎯 Research Background / 研究背景
Termites are destructive pests that threaten buildings, water conservancy projects and cultural relics, causing global economic losses of up to 40 billion USD annually. With global warming, the activity range of termites continues to expand.

1. Traditional monitoring methods such as manual inspection have the disadvantages of high cost, slow response and limited coverage;
2. Most existing intelligent monitoring devices only support qualitative judgment of termite activity, and cannot quantify pest population and infestation level;
3. Termite metabolism produces a large amount of $CO_2$ and $CH_4$, and gas concentration is positively correlated with termite quantity, but quantitative prediction under complex environmental interference is insufficient;
4. Conventional time series models cannot effectively balance local feature extraction, global dependency modeling and multi-factor nonlinear coupling.

This study takes Zhuji City, Zhejiang Province as the research area, builds a complete monitoring and data analysis pipeline, and proposes a parallel deep learning model to achieve accurate classification of termite infestation levels.

---

## 🛠️ System & Data Acquisition / 采集系统与数据集
### 1. Monitoring Platform / 监测硬件平台
We designed a sealed termite gas monitoring device, including sealed experimental chamber, gas sensor module, temperature & humidity sensor and data acquisition unit.
- Volume of experimental box: 5.5 L
- Sampling frequency: 1 record per minute
- Continuous monitoring duration for each group: 48 hours

<div align="center">
<!-- 此处放置设备结构图/实物图 -->
<img src="assets/device_overview.png" width="700" alt="Monitoring Platform & Experimental Setup">
<p><i>Figure 1. Experimental setup and multi-node termite monitoring system</i></p>
</div>

### 2. Data Preprocessing Pipeline / 数据预处理流程
1. **Raw Data**: 272,230 original time-series records, covering termite populations from 50 to 900 individuals;
2. **Data Filtering**: Remove missing values, duplicate records and samples with termite mortality over 5%;
3. **Outlier Removal**: Adopt sliding window median + MAD method to eliminate sensor noise;
4. **Sliding Window**: Use **10-minute non-overlapping window** to extract statistical features (mean, max, min, std, variation, slope);
5. **Final Dataset**: 26,820 valid structured samples, each sample is a 26-dimensional feature vector.

<div align="center">
<!-- 此处放置数据处理流程图 -->
<img src="assets/data_pipeline.png" width="700" alt="Data Processing Workflow">
<p><i>Figure 2. Overall workflow of data collection and feature engineering</i></p>
</div>

### 3. Dataset Partition / 数据集划分
To avoid data leakage, the dataset is divided by experimental batches instead of single samples:
- Train Set: 62 batches / 18,774 samples (70%)
- Validation Set: 9 batches / 2,682 samples (10%)
- Test Set: 18 batches / 5,364 samples (20%)

### 4. Infestation Level Classification / 虫害等级划分
Referring to Chinese national standard GB/T 51253-2017, termite infestation is divided into three risk levels:

| Risk Level | Termite Quantity | Label | Hazard Degree | Suggested Measures |
| :--------- | :--------------- | :---- | :------------ | :---------------- |
| Low Risk   | ≤ 100            | 0     | Mild          | Regular monitoring |
| Medium Risk| 101 ~ 499        | 1     | Moderate      | Local treatment |
| High Risk  | ≥ 500            | 2     | Severe        | Nest excavation & full-area control |

### 5. Data Distribution / 数据分布
| Risk Level | Corresponding Groups | Total Samples | Proportion | Test Samples |
| :--------- | :------------------- | :------------ | :--------- | :----------- |
| Low Risk   | 50, 100              | 6027          | 22.47%     | 1205         |
| Medium Risk| 200, 300, 400        | 9040          | 33.71%     | 1808         |
| High Risk  | 500 ~ 900            | 11753         | 43.82%     | 2351         |
| **Total**  | —                    | **26820**     | **100%**   | **5364**     |

### 6. Data Permission Statement / 数据权限说明
All experimental data used in this study are sourced from local government departments in China, which involves confidential information. **The original dataset cannot be publicly shared, distributed or provided externally**.

---

## 🧠 Model Architecture / PAFTI-Net 模型结构
PAFTI-Net follows the design logic: `Adaptive Feature Enhancement → Multi-path Parallel Feature Extraction → Dynamic Fusion → Classification`.

The whole network consists of four core modules:
1. **AFFM (Adaptive Feature Fusion Module)**: Dynamically weight input features, enhance valid signals and suppress noise;
2. **PPEM (Parallel Path Extraction Module)**: Three parallel branches for multi-scale local features, global time dependencies and factor interaction features;
   - MCP: Multi-scale 1D Convolution Path (capture local fluctuations)
   - CFM: Cross-Time Fusion Module (Transformer-based global dependency modeling)
   - FIP: Factor Interaction Path (mine nonlinear coupling between multiple factors)
3. **DPFM (Dynamic Path Fusion Module)**: Adaptively fuse features from three branches;
4. **Classification Head**: Temporal attention pooling + fully connected layers to output infestation level.

<div align="center">
<!-- 此处放置模型整体结构图 -->
<img src="assets/model_architecture.png" width="800" alt="PAFTI-Net Architecture">
<p><i>Figure 3. Overall architecture of the proposed PAFTI-Net</i></p>
</div>

<div align="center">
<!-- 此处放置分类头细节图 -->
<img src="assets/classifier_head.png" width="500" alt="Classification Module">
<p><i>Figure 4. Structure of classification module</i></p>
</div>

---

## ⚙️ Experimental Environment / 实验环境
| Item | Configuration |
| :--- | :------------ |
| OS | Windows 10 Professional |
| CPU | Intel Xeon Gold 5218R @ 2.10 GHz |
| GPU | NVIDIA RTX 3090 |
| Python | 3.9 |
| Deep Learning Framework | PyTorch 2.0 |
| Batch Size | 16 |
| Epochs | 200 |
| Optimizer | Adam (lr = 0.001) |
| Dropout Rate | 0.5 |
| Input Sequence Length | 5 |

---

## 📈 Experimental Results / 实验结果
### 1. Comparison with Baselines / 主流模型对比
We select DeepAR, 1D CNN-LSTM and ESD-TripleStream as baseline models. PAFTI-Net achieves the best performance on all metrics.

<div align="center">
<!-- 此处放置结果对比柱状图 -->
<img src="assets/result_comparison.png" width="750" alt="Model Performance Comparison">
<p><i>Figure 5. Quantitative results of different models</i></p>
</div>

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-score (%) |
| :---- | :----------- | :------------ | :--------- | :---------- |
| 1D CNN-LSTM | 81.97 | 78.64 | 80.95 | 82.56 |
| DeepAR | 84.24 | 84.46 | 82.85 | 80.95 |
| ESD-TripleStream | 90.12 | 89.45 | 82.56 | 84.46 |
| **PAFTI-Net (Ours)** | **92.31** | **89.45** | **90.01** | **89.71** |

### 2. Ablation Study / 消融实验
Verify the effectiveness of each core module of PAFTI-Net:

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-score (%) |
| :---- | :----------- | :------------ | :--------- | :---------- |
| w/o AFFM | 91.77 | 89.06 | 89.19 | 89.09 |
| w/o PPEM | 84.79 | 80.79 | 80.14 | 80.18 |
| w/o DPFM | 90.88 | 88.83 | 86.79 | 87.55 |
| w/o MCP | 88.37 | 85.15 | 82.77 | 83.34 |
| w/o CFM | 90.16 | 88.55 | 86.24 | 87.10 |
| w/o FIP | 88.37 | 85.94 | 84.12 | 84.75 |
| **PAFTI-Net** | **92.31** | **89.45** | **90.01** | **89.71** |

### 3. Time Step & Feature Analysis / 时序步长 & 特征分析
- Optimal sliding window time step: **10 steps (10 minutes)**
- Core features: $CO_2$ and $CH_4$ related statistical features have the highest importance;
- Optimal input dimension: Top 15 high-importance features.

### 4. Class-wise Performance / 分级识别效果
The model performs best on high-risk samples, and maintains stable recognition ability for low & medium risk, meeting the actual early warning requirements of pest control.

<div align="center">
<!-- 此处放置分级性能图 -->
<img src="assets/class_performance.png" width="650" alt="Performance for each risk level">
<p><i>Figure 6. Classification performance on three infestation levels</i></p>
</div>

---

## ✅ Conclusion / 总结
1. A complete multi-sensor synchronous monitoring platform and standardized termite gas time-series dataset are constructed;
2. The proposed PAFTI-Net combines adaptive feature fusion and parallel multi-branch structure, which effectively fuses local, global and interactive features of time series;
3. Experiments prove that PAFTI-Net outperforms existing time-series models, and can be applied to real-time intelligent monitoring and graded early warning of termites.

### Future Work / 未来工作
- Deploy the model to multi-site field monitoring terminals;
- Introduce domain adaptation and meta-learning to improve model robustness in complex field environments;
- Expand research on metabolic gas characteristics of multiple pest species.

---

## 📚 Citation / 引用格式
If you use this work for your research, please cite our paper:
```bibtex
@article{wu2024paftinet,
  title={PAFTI-Net: A Parallel Deep Learning Model for Termite Infestation Level Prediction Using Gas Monitoring Data},
  author={Wu, Shan and Wang, Hangjun and Lu, Weiling},
  journal={IEEE Access},
  volume={11},
  year={2023},
  doi={10.1109/ACCESS.2024.0429000}
}

