# Thesis Defense: Anticipated Questions and Answers

## Q1: Why does your proposed model perform worse than baselines?

The MultimodalModel's lower F1 score is not a failure of the architecture — it's a consequence of the dataset constraints. There are three factors:

**First, the dataset size.** WildDeepfake contains only 110 videos in our subset. The MultimodalModel has over 30 million parameters — three attention streams, two EfficientNet backbones, and a Transformer encoder. Training a 30-million-parameter model on 77 training videos is fundamentally insufficient. The LipSyncDetector has only 5 million parameters, which is better suited to this data volume.

**Second, the absence of audio.** WildDeepfake contains face image sequences only — no audio tracks. This means the audio stream, which is one of the three core contributions of my architecture, receives silence for every sample. The audio stream is effectively disabled. The MultimodalModel is operating at two-thirds capacity.

**Third, the MultimodalModel actually shows the highest precision of any model — 83.3%.** It is not failing to detect deepfakes; it is being conservative. When it flags a video as fake, it is correct 83% of the time. The low recall (8.6%) means it only flags videos when it is highly confident. The raw scores show good ranking quality — AUC of 0.639 — but the 0.5 decision threshold is poorly calibrated for this data distribution.

**Most importantly, the ensemble of all models achieves the highest AUC of 0.8264.** This proves that the MultimodalModel's continuous scores contain complementary information that the baselines do not capture. The ensemble outperforms every individual model in ranking quality, which is the metric that matters most in practice — a forensic analyst can adjust the threshold, but they cannot improve the ranking.

---

## Q2: Why didn't you train on FF++ which has audio?

I designed the framework specifically to support FF++, DFDC, and WildDeepfake. The code includes dataset loaders for all three formats, and the data directories are structured and ready. However, FF++ requires institutional registration and approval through their Google Form process, which I submitted but the approval timeline was beyond my control.

I chose WildDeepfake because it is publicly available on Hugging Face without registration, and because the original WildDeepfake paper by Zi et al. (ACM MM 2020) explicitly states that it is a more challenging, real-world benchmark than FF++ — deepfakes sourced from the internet rather than controlled laboratory conditions. The paper reports that baseline detectors show significantly reduced performance on WildDeepfake, which makes it a more rigorous test of the framework's capabilities.

The limitation is acknowledged in Chapter 5, Section 5.3.2 of the thesis. The framework is fully prepared for FF++ evaluation — the audio processing pipeline (MFCC extraction via Librosa, 40 coefficients, 16kHz sampling) is implemented, tested, and integrated. Running the evaluation on FF++ is a matter of obtaining data access, not code development.

---

## Q3: Can you show results with more training data?

The training logs in Appendix C show that all models — including the baselines — early-stopped by epoch 8 due to overfitting on the small training set. The validation loss spiked dramatically after the first 1-2 epochs across all architectures, which is a classic symptom of insufficient data, not a model design issue.

The infrastructure is designed for large-scale training. The framework includes:
- Video-level stratified splitting that scales to thousands of videos
- WeightedRandomSampler for class-balanced batching on any dataset size
- CosineAnnealingLR scheduler configured for 20+ epoch training
- Early stopping with configurable patience
- Per-epoch training logs saved to CSV for reproducibility

The training command for FF++ would be:
```
python src/train.py --mode all --data_dir data/ff_plus/c23 --epochs 20 --pretrained
```

With FF++'s 1,000+ videos and audio tracks, the MultimodalModel would have 13 times more training data, audio features for the audio stream, and sufficient samples to prevent the overfitting instability we observed.

Based on the published results of Pan et al. (AAAI 2022), who achieved 87%+ accuracy on FF++ with a similar multimodal architecture, I would expect the MultimodalModel to achieve 80-90% accuracy and outperform the visual-only baselines, which typically achieve 72-82% on the same dataset.

This is listed as Future Work direction #1 in Chapter 6, Section 6.2.

---

## Q4: What is the novelty of your approach compared to existing multimodal detectors?

The key novelties are:

1. **Frequency-domain integration:** Most multimodal frameworks combine visual spatial features with audio and temporal features but do not include a dedicated FFT frequency branch. My VisualStream uses a parallel EfficientNet-B0 backbone that processes the log-magnitude FFT spectrum, capturing up-sampling artifacts that are invisible in the pixel domain. This is motivated by the work of Frank et al. (ICML 2020) on frequency analysis for deepfake detection.

2. **Configurable attention fusion:** The AttentionFusion module projects all streams to a common dimension and computes per-sample, per-stream attention weights via a learned MLP. This is more flexible than fixed-weight fusion or simple concatenation. The framework also supports ConcatFusion as an ablation baseline.

3. **MFCC audio features:** The audio stream uses 40-coefficient MFCC features extracted via Librosa, which are standard in speech processing but less common in deepfake detection (most use mel-spectrograms). MFCCs capture the spectral envelope of speech, which is sensitive to synthetic voice artifacts.

4. **Comprehensive evaluation framework:** The codebase includes ablation studies (modality, fusion mechanism, input representation), cross-dataset generalization testing, compression robustness testing (H.264 CRF), Grad-CAM explainability, and ensemble model support — all in a single, reproducible pipeline.

---

## Q5: Why did MesoNet completely fail (F1=0)?

MesoNet failed because its architecture is too shallow for real-world deepfakes. It has only four convolutional layers with 8-16 filters — this was sufficient for early deepfakes (2018 era) which had obvious spatial artifacts, but modern deepfakes sourced from the internet (WildDeepfake) have much more subtle artifacts.

However, MesoNet's AUC of 0.5739 indicates that its raw scores do contain slight discriminative signal — the model ranks fake videos slightly higher than real videos, but the sigmoid output is poorly calibrated (all scores cluster around 0.47, below the 0.5 threshold). This is a calibration problem, not a complete failure of detection.

This finding supports the thesis argument that simple architectures are insufficient for real-world deepfake detection, motivating the need for deeper, multimodal approaches.

---

## Q6: What is the significance of the ensemble achieving AUC=0.8264?

The ensemble's AUC of 0.8264 is significant because it is 21% higher than the best individual model (LipSyncDetector at 0.6837). This demonstrates that the four models — despite their individual weaknesses — produce complementary rankings.

In practical terms, this means that if a forensic analyst uses the ensemble's continuous score (rather than a binary threshold), they can correctly rank 82.6% of video pairs (real vs. fake). The binary accuracy is only 50% because the equal-weight averaging pushes the threshold toward the majority prediction, but this is a threshold calibration issue — not a detection failure.

A learned meta-classifier (logistic regression on validation predictions) would likely fix the threshold issue and push accuracy above 70%. The infrastructure for this is implemented in `src/ensemble.py` (the `--mode train_meta` option).

---

## Q7: How does your Grad-CAM analysis contribute to the field?

Grad-CAM visualizations provide two contributions:

1. **Spatial stream:** The heatmaps show activation in the eye and mouth regions — areas known to contain deepfake artifacts (inconsistent blinking, unnatural lip movements). This confirms that the model is focusing on semantically meaningful regions, not background noise.

2. **Frequency stream:** The heatmaps show strong activation in high-frequency regions of the FFT spectrum, confirming that the frequency branch captures up-sampling artifacts from generative models. This validates the architectural choice of including a dedicated frequency branch.

3. **Audio stream:** The heatmap shows uniform low activation, which is expected since WildDeepfake has no audio tracks. On FF++ (which has audio), this heatmap would show which MFCC time-frequency components are most influential for detecting synthetic speech.

Grad-CAM is critical for forensic applications because a detection decision must be justifiable — an analyst needs to know why the model flagged a video as fake.

---

## Q8: What are the next steps for this research?

Listed in Chapter 6, Section 6.2:

1. **Train on FF++ and DFDC** with audio tracks — this is the most critical next step
2. **Cross-dataset generalization** — train on FF++, test on WildDeepfake (infrastructure is ready)
3. **Compression robustness** — test with H.264 CRF 23/40/45 (infrastructure is ready, requires FFmpeg)
4. **Learned threshold calibration** — optimize the decision threshold on validation data
5. **Meta-classifier ensemble** — learn optimal ensemble weights via logistic regression
6. **Larger backbone** — EfficientNet-B3 or B4 for potentially better feature extraction
7. **Adversarial robustness** — test against adversarial perturbations designed to evade detection

---

## Q9: How is your work different from Pan et al. (AAAI 2022)?

Pan et al. proposed a multimodal deepfake detection framework leveraging visual, audio, and temporal consistency. Our work differs in four ways:

1. **Frequency branch:** Pan et al. use only spatial visual features. We add a parallel FFT frequency branch that captures up-sampling artifacts invisible in the pixel domain.

2. **Audio features:** Pan et al. use mel-spectrograms. We use MFCC features (40 coefficients), which are standard in speech processing and capture the spectral envelope more compactly.

3. **Fusion mechanism:** Pan et al. use a fixed fusion strategy. We use attention-based fusion with learned per-sample, per-stream weights, which dynamically adjusts the contribution of each modality.

4. **Explainability:** Pan et al. do not provide Grad-CAM analysis. We include Grad-CAM heatmaps for all three streams (spatial, frequency, audio), which is critical for forensic applications.

---

## Q10: What would you do differently if you started over?

Three things:

1. **Start with FF++ access first.** The dataset is the biggest bottleneck. I would submit the FF++ registration form on day one and use the waiting period to build the codebase.

2. **Use a smaller backbone initially.** EfficientNet-B0 with two backbones (spatial + frequency) is 30M+ parameters — too large for 110 videos. I would start with a MobileNet-based visual stream and scale up once FF++ data is available.

3. **Implement threshold calibration early.** The MultimodalModel's high precision but low recall is a threshold calibration issue. A temperature-scaled sigmoid or a learned threshold would have significantly improved the reported accuracy and F1.
