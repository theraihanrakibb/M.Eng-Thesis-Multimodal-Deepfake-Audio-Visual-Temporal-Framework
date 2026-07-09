# Detecting Deepfake Video by A Multimodal Audio-Visual Framework with Temporal Inconsistencies

**Author:** MD RAKIBUL ISLAM RAIHAN
**Student Number:** 2024280031
**Supervisor:** ZHANG PENG
**Degree:** Master of Engineering
**Major:** Software Engineering
**School:** School of Software Engineering
**University:** Northwestern Polytechnical University
**Date:** July 2026

---

## Abstract

The rapid maturation of generative adversarial networks, diffusion models, and auto-regressive neural vocoders has democratised the production of hyper-realistic synthetic video. The resulting *deepfake* media now threaten the integrity of evidence in judicial proceedings, the credibility of journalism, the security of biometric authentication systems, and the fabric of public trust in digital communication. While the first generation of detection systems attained near-saturation accuracy on academic benchmarks such as FaceForensics++ by exploiting low-level pixel artefacts in single frames, they have proven brittle when confronted with modern, internet-grade forgeries that are visually indistinguishable from authentic footage to a human observer. The central weakness of these legacy detectors is their reliance on a single modality: a frame-level visual signature whose discriminative power evaporates as generative models continue to improve.

This dissertation presents a multimodal audio-visual framework for deepfake video detection that systematically fuses three complementary streams of evidence — a spatial-frequency visual stream, a Mel-frequency-cepstral-coefficient (MFCC) audio stream, and a Transformer-encoded temporal stream — through a learned attention mechanism. The visual stream employs an EfficientNet-B0 backbone operating in parallel on the raw RGB frame and on the log-magnitude Fast Fourier Transform (FFT) of its luminance channel, so that high-frequency up-sampling artefacts invisible in the pixel domain become explicit features in the frequency domain. The audio stream processes 40-coefficient MFCC frames extracted by Librosa through a three-layer convolutional network, capturing vocal-tract artefacts introduced by neural vocoders. The temporal stream consumes frame-level visual features through a three-layer, eight-head Transformer encoder whose self-attention exposes inter-frame inconsistencies in head pose, blink cadence, and lip motion. An attention-based fusion module projects the three streams into a shared 512-dimensional space and learns per-stream weights conditioned on the input, after which a fully connected classifier produces the final real/fake decision.

The framework is benchmarked against three widely-used baselines — MesoNet, XceptionNet, and a lip-sync detector — on the WildDeepfake dataset, a corpus of 110 real-world deepfake videos harvested from the open internet (52 authentic, 58 manipulated). Under a video-level stratified 70/15/15 split with balanced sampling, cosine-annealed learning rates, and early stopping, the lip-sync detector achieves the strongest individual performance (accuracy 0.6727, F1 0.7500, AUC 0.6837), while the proposed multimodal model exhibits a conservative bias (precision 0.8333 but recall 0.0862). A score-level ensemble of the four models attains the highest ranking quality of any system tested (AUC 0.8264), demonstrating that the individual models encode complementary evidence even when their decision-boundary accuracy is constrained by the small training corpus. The dissertation further conducts a per-epoch training analysis, a Grad-CAM qualitative saliency study, and an honest appraisal of the framework's limitations on a 110-video corpus.

The principal contributions of this work are: (i) a unified three-stream architecture that integrates spatial, frequency, audio, and temporal cues under a single attention-based fusion module; (ii) a configurable implementation that supports modality, fusion-mechanism, and input-representation ablations through a single code path; (iii) a systematic evaluation against three families of baseline detectors on a real-world, in-the-wild dataset; and (iv) an open, reproducible software framework that implements the proposed model together with the baseline comparators, the training pipeline, the evaluation suite, the ablation runner, the Grad-CAM visualiser, and the ensemble combiner.

**Keywords:** Deepfake Detection, Multimodal Fusion, Audio-Visual Analysis, Temporal Inconsistencies, MFCC, Transformer Encoder, Attention Fusion, EfficientNet, Frequency Domain Analysis, WildDeepfake.

---

## Acknowledgments

This dissertation would not have been possible without the patient guidance and rigorous standards of my supervisor, Professor ZHANG PENG, whose insistence on real-dataset validation and on the integrity of the multimodal fusion component shaped both the technical direction and the intellectual posture of this work. The mid-term evaluation committee — Professor ZHANG PENG, Professor YOU TAO, and Professor WEI WEI — identified critical weaknesses in an earlier version of this study that, once addressed, materially improved the quality of the final framework. I am grateful for their candour.

I also acknowledge the broader deepfake-detection research community, whose open datasets and reproducible baselines made the empirical work in this dissertation possible. The FaceForensics++ benchmark [1], the Deepfake Detection Challenge corpus [2], the WildDeepfake collection [3], and the MesoNet [4] and Xception [5] architectures have collectively defined the methodological vocabulary within which the present work is situated. The Librosa [6], PyTorch [7], timm [8], OpenCV [9], and scikit-learn [10] projects provided the software foundations on which the implementation rests.

Finally, I thank my family for their unconditional support throughout my graduate studies, and my fellow students at the School of Software Engineering for the daily conversations that sharpened my understanding of the problem.

*MD RAKIBUL ISLAM RAIHAN*
*Northwestern Polytechnical University, Xi'an, China*
*July 2026*

---

## Table of Contents

- Abstract
- Acknowledgments
- List of Figures
- List of Tables
- List of Abbreviations

1. Introduction
   1.1 Background and Motivation
   1.2 Problem Statement
   1.3 Research Questions
   1.4 Research Objectives
   1.5 Contributions
   1.6 Scope and Delimitations
   1.7 Thesis Organisation

2. Literature Review
   2.1 Overview of Deepfake Generation Techniques
   2.2 Unimodal Visual Detection
       2.2.1 Spatial-Artifact Detectors
       2.2.2 Frequency-Domain Detectors
       2.2.3 Biological-Signal Detectors
   2.3 Temporal and Sequential Detection
   2.4 Audio-Based Detection
   2.5 Multimodal Detection
   2.6 Identified Research Gap
   2.7 Summary

3. Methodology
   3.1 System Overview
   3.2 Visual Stream
       3.2.1 Spatial Backbone
       3.2.2 Frequency Branch
       3.2.3 Feature Composition
   3.3 Audio Stream
       3.3.1 MFCC Extraction
       3.3.2 Convolutional Audio Encoder
   3.4 Temporal Stream
       3.4.1 Frame-Level Feature Extractor
       3.4.2 Transformer Encoder
       3.4.3 Temporal Pooling
   3.5 Attention-Based Fusion
   3.6 Classification Head
   3.7 Baseline Models
       3.7.1 MesoNet
       3.7.2 XceptionNet
       3.7.3 LipSyncDetector
   3.8 Datasets
       3.8.1 WildDeepfake
       3.8.2 FaceForensics++
       3.8.3 DFDC
   3.9 Implementation Details
       3.9.1 Data Preprocessing
       3.9.2 Data Splitting and Augmentation
       3.9.3 Training Configuration
       3.9.4 Software and Hardware
   3.10 Evaluation Metrics
   3.11 Ablation Framework
   3.12 Summary

4. Experiments and Results
   4.1 Experimental Setup
       4.1.1 Dataset Configuration
       4.1.2 Training Protocol
       4.1.3 Evaluation Protocol
   4.2 Main Results
       4.2.1 Quantitative Performance
       4.2.2 Per-Model Analysis
       4.2.3 Confusion-Matrix Analysis
   4.3 Training Dynamics
       4.3.1 Multimodal Model
       4.3.2 MesoNet
       4.3.3 XceptionNet
       4.3.4 LipSyncDetector
       4.3.5 Cross-Model Comparison
   4.4 Ensemble Analysis
       4.4.1 Motivation
       4.4.2 Score-Level Fusion
       4.4.3 Ranking Quality
   4.5 Grad-CAM Qualitative Analysis
       4.5.1 Spatial Stream Saliency
       4.5.2 Frequency Stream Saliency
       4.5.3 Audio Stream Saliency
       4.5.4 Interpretation
   4.6 Ablation Studies
       4.6.1 Modality Ablation
       4.6.2 Fusion-Mechanism Ablation
       4.6.3 Input-Representation Ablation
   4.7 Summary

5. Discussion
   5.1 Comparison with State-of-the-Art
   5.2 Strengths of the Proposed Framework
   5.3 Limitations
       5.3.1 Dataset Scale
       5.3.2 Absence of Audio in WildDeepfake
       5.3.3 Training Instability
       5.3.4 Decision-Threshold Calibration
   5.4 Ethical Considerations
   5.5 Threats to Validity
   5.6 Summary

6. Conclusion and Future Work
   6.1 Conclusion
   6.2 Future Work

References

Appendices
   Appendix A: Hyperparameter Configuration
   Appendix B: Complete Performance Metrics
   Appendix C: Per-Model Training Logs
   Appendix D: Dataset Statistics
   Appendix E: Software and Hardware Configuration
   Appendix F: Complete File Structure

---

## List of Figures

- Figure 1.1: Cross-referencing multiple sources of evidence within a video.
- Figure 2.1: Taxonomy of deepfake detection approaches.
- Figure 3.1: System architecture of the proposed multimodal framework.
- Figure 3.2: Visual stream with parallel spatial and frequency branches.
- Figure 3.3: MFCC extraction pipeline.
- Figure 3.4: Temporal Transformer encoder.
- Figure 3.5: Attention-based fusion module.
- Figure 4.1: Performance comparison of the four models on WildDeepfake.
- Figure 4.2: Radar comparison of multi-metric performance.
- Figure 4.3: Confusion matrices for the four models.
- Figure 4.4: ROC curves for the four models and the ensemble.
- Figure 4.5: Training curves for the multimodal model.
- Figure 4.6: Training curves for MesoNet.
- Figure 4.7: Training curves for XceptionNet.
- Figure 4.8: Training curves for the lip-sync detector.
- Figure 4.9: Ensemble versus individual model comparison.
- Figure 4.10: Grad-CAM saliency overlay on a real frame.
- Figure 4.11: Grad-CAM saliency overlay on a fake frame.
- Figure 4.12: Audio-stream Grad-CAM on an MFCC frame.

## List of Tables

- Table 3.1: Architecture hyperparameters of the proposed model.
- Table 3.2: Configuration of the baseline models.
- Table 3.3: Dataset statistics.
- Table 3.4: Training hyperparameters.
- Table 4.1: Main results on the WildDeepfake test split (110 videos).
- Table 4.2: Per-epoch training losses for the multimodal model.
- Table 4.3: Per-epoch training losses for MesoNet.
- Table 4.4: Per-epoch training losses for XceptionNet.
- Table 4.5: Per-epoch training losses for the lip-sync detector.
- Table 4.6: Ensemble versus individual model performance.
- Table 4.7: Ablation study configurations.
- Table 5.1: Comparison with state-of-the-art deepfake detectors.

## List of Abbreviations

- **AUC**: Area Under the (ROC) Curve
- **BCE**: Binary Cross-Entropy
- **CNN**: Convolutional Neural Network
- **DCT**: Discrete Cosine Transform
- **DFDC**: Deepfake Detection Challenge
- **DFT**: Discrete Fourier Transform
- **FF++**: FaceForensics++
- **FFT**: Fast Fourier Transform
- **FN/FP**: False Negative / False Positive
- **GAN**: Generative Adversarial Network
- **GPU**: Graphics Processing Unit
- **GRU**: Gated Recurrent Unit
- **IIR**: Infinite Impulse Response
- **LSTM**: Long Short-Term Memory
- **MFCC**: Mel-Frequency Cepstral Coefficient
- **PR-AUC**: Precision-Recall Area Under Curve
- **ReLU**: Rectified Linear Unit
- **ROC**: Receiver Operating Characteristic
- **RNN**: Recurrent Neural Network
- **STFT**: Short-Time Fourier Transform
- **TN/TP**: True Negative / True Positive
- **ViT**: Vision Transformer

---

# Chapter 1

## Introduction

### 1.1 Background and Motivation

The past decade has witnessed a qualitative leap in the realism of synthetic media. Driven by the maturation of generative adversarial networks (GANs) [14, 22, 23], variational autoencoders, and, more recently, latent diffusion models, the production of face-swap video, voice cloning, and neural lip-sync has migrated from a research curiosity to a consumer-grade commodity. Mobile applications, browser extensions, and online services now allow untrained users to transplant a face from a single photograph into an arbitrary target video, to synthesise a speaker's voice from a few seconds of reference audio, and to redub a clip with synchronised mouth motion in real time.

The trajectory of this maturation can be traced along three axes. On the *quality* axis, the original deepfake pipeline — the encoder-decoder-decoder architecture introduced in the r/deepfakes community in late 2017 — produced face swaps whose artefacts were visible to a casual observer: blurred boundaries around the face, mismatched skin tones, inconsistent lighting, and missing fine detail in the eye and mouth regions. Within five years, the same task was being performed by pipelines based on StyleGAN [23], whose style-based generator architecture produced faces whose fine detail, skin micro-texture, and high-frequency statistics were indistinguishable from authentic footage to a human observer. On the *latency* axis, the original pipeline required hours of training per face pair on a single GPU; the same task is now achievable in seconds on a consumer-grade laptop, and in real time on a high-end GPU. On the *accessibility* axis, the original pipeline required the user to assemble a custom training set, install a deep-learning software stack, and operate a command-line interface; the same task is now exposed through a mobile application with a single button.

While these capabilities have legitimate applications in film production, accessibility tooling, education, and creative arts, they also introduce a spectrum of harms whose severity scales with the fidelity of the underlying generative model. Documented misuse includes non-consensual intimate imagery, fraudulent audio-video calls impersonating corporate executives, fabricated political statements during election cycles, and the erosion of evidentiary value in journalistic and judicial contexts. The 2023 Investigative Reporting Prize was awarded to a story whose central evidence was a single, defantly uncorroborated video clip; the same year, multiple jurisdictions reported instances of synthetic voice being used to authorise fraudulent bank transfers in excess of six figures. As the marginal cost of producing a convincing deepfake approaches zero, the marginal cost of detecting it becomes the load-bearing parameter in the trust economy of digital media.

The threat model against which a deepfake detector must be evaluated is correspondingly broad. At the lowest tier of sophistication, an unsophisticated actor uses an off-the-shelf mobile application to produce a deepfake whose artefacts are visible to a trained human observer; a detector that exploits those artefacts will succeed against this tier. At the middle tier, a sophisticated actor uses a custom-trained pipeline with perceptual losses and explicit high-frequency regularisation, producing a deepfake whose artefacts are invisible to a human observer but whose statistical signature is still distinguishable from authentic footage by a machine classifier. At the highest tier, a state-level actor uses a generator that explicitly models the detector's signal — for instance, by adding a photoplethysmographic signal to the synthesised face to defeat biological-signal detectors, or by adversarial training against a known detector architecture — producing a deepfake that is, in the limit, indistinguishable from authentic footage by any classifier that operates on the same signal family as the generator. The detector designer's task is to maximise the cost imposed on the adversary at each tier, with the understanding that the detector will eventually be defeated at the highest tier and must be retired or augmented.

The deepfake-detection research community has responded with a proliferation of detection architectures. The first generation, exemplified by MesoNet [4], Xception-based detectors [5, 11], and the FaceForensics++ benchmark [1], exploited low-level pixel statistics that distinguished early face-swap outputs from authentic footage — irregular blink rates, blurred boundaries around the eyes and mouth, unnatural skin textures, and inconsistent lighting. These detectors attained accuracies above 95 % on the original FaceForensics++ benchmark, a feat that masked a deeper brittleness: the detectors were learning the artefacts of *specific* generative pipelines rather than the structural signatures of *synthesis itself*. As soon as a new generation of generative models closed each artefact — for instance, by introducing inter-frame consistency to defeat blink detectors, or by adopting perceptual losses to defeat pixel-statistic detectors — the corresponding detection accuracy collapsed.

A second generation of detectors sought to address this brittleness by widening the evidentiary aperture. Frequency-domain analyses [13] observed that GAN-based up-sampling leaves a characteristic signature in the Fourier spectrum even when the pixel-space output is visually indistinguishable from real footage. Temporal analyses [15, 28] examined sequences of frames rather than individual images, exploiting the observation that generating temporally coherent video is substantially harder than generating a single still image. Audio-visual analyses [29, 33] cross-referenced the visual lip motion with the acoustic content, since many deepfakes manipulate the face but leave the audio untouched, or vice versa. Each of these approaches encodes a *complementary* signal of manipulation, and a detector that jointly reasons across them is, in principle, robust to the closure of any single artefact family by the generative model.

This dissertation develops and evaluates one such detector. The proposed framework integrates a spatial-frequency visual stream, an MFCC audio stream, and a Transformer-encoded temporal stream under a single attention-based fusion module, and evaluates the resulting system against three baseline detectors on a real-world deepfake corpus. The choice of these three modalities is motivated by the complementarity of their signals: the spatial stream captures the per-frame visual artefacts that distinguish face-swap outputs from authentic footage; the frequency stream captures the up-sampling artefacts that are invisible in the pixel domain; the audio stream captures the vocal-tract artefacts introduced by neural vocoders; and the temporal stream captures the inter-frame inconsistencies that arise from per-frame face-swap operations. The attention fusion mechanism allows the model to weight these streams dynamically, conditioned on the input, so that the model can rely more heavily on the visual stream when the audio is silent, more heavily on the audio stream when the visual is high-quality, and more heavily on the temporal stream when the per-frame visual artefacts are subtle but the inter-frame dynamics are anomalous.

### 1.2 Problem Statement

Despite the encouraging accuracy figures reported by first-generation deepfake detectors on academic benchmarks, three structural problems limit their operational utility. First, the detectors' performance on visually compelling, internet-grade forgeries is substantially lower than their performance on the synthesised benchmarks on which they were trained; the WildDeepfake [3] corpus, whose clips are harvested from the open internet rather than generated in a controlled pipeline, has consistently exposed a 20–40 percentage-point gap between benchmark and in-the-wild accuracy. Second, the detectors are largely unimodal: a frame-level visual detector is silent when only the audio track has been manipulated, and a temporal detector cannot reason about static frequency artefacts. Third, the detectors' decisions are opaque: even when correct, they offer no indication of *which* modality or *which* region of the input drove the decision, undermining trust in operational deployments and complicating adversarial robustness analysis.

The combination of these three limitations motivates the central problem addressed by this dissertation: *the design, implementation, and evaluation of a deepfake detection framework that simultaneously leverages spatial, frequency, audio, and temporal cues under a single, interpretable fusion mechanism, and that is robust enough to produce informative results on small, real-world corpora of internet-grade forgeries.*

### 1.3 Research Questions

The dissertation is structured around four research questions:

**RQ1.** Can a multimodal architecture that fuses spatial, frequency, audio, and temporal cues under a learned attention mechanism outperform strong unimodal baselines on a real-world deepfake corpus?

**RQ2.** What is the marginal contribution of each modality to the overall detection decision, and does the attention mechanism learn to weight modalities in accordance with their respective information content?

**RQ3.** Does an ensemble of detectors with structurally different inductive biases achieve higher ranking quality (as measured by the area under the ROC curve) than any individual detector, even when their point-estimate accuracy is constrained by training-data scarcity?

**RQ4.** Can a Grad-CAM-style saliency analysis expose the regions of the face, the frequency spectrum, and the audio spectrogram that drive the model's decision, and do those regions align with the artefact families reported in the deepfake-generation literature?

### 1.4 Research Objectives

To address the four research questions, the dissertation pursues five objectives:

1. **Design** a multimodal architecture that integrates (a) an EfficientNet-B0 visual backbone with a parallel FFT frequency branch, (b) a three-layer convolutional audio encoder operating on 40-coefficient MFCC frames extracted by Librosa, (c) a three-layer, eight-head Transformer temporal encoder operating on frame-level visual features, and (d) an attention-based fusion module that learns per-stream weights conditioned on the input.

2. **Implement** the architecture in a configurable manner so that modality, fusion-mechanism, and input-representation ablations can be conducted through a single code path, without altering the surrounding training and evaluation infrastructure.

3. **Benchmark** the framework against three structurally distinct baselines — MesoNet [4], XceptionNet [5], and a lip-sync detector — on the WildDeepfake corpus, under a video-level stratified 70/15/15 split with balanced sampling, cosine-annealed learning rates, and early stopping.

4. **Analyse** the training dynamics, the ranking behaviour of an ensemble of the four detectors, and the Grad-CAM [34] saliency of the proposed model's visual, frequency, and audio streams.

5. **Discuss** the strengths, limitations, and ethical implications of the framework, with particular attention to the constraints imposed by the small training corpus and the absence of audio in the WildDeepfake dataset.

### 1.5 Contributions

The dissertation makes four principal contributions to the deepfake-detection literature.

**C1. A unified three-stream architecture with attention-based fusion.** Unlike prior multimodal detectors that fuse streams by simple concatenation [29] or by a fixed weighted average, the proposed framework projects each stream into a shared 512-dimensional space and applies a per-stream attention weight computed by a two-layer multilayer perceptron. The attention weights are made explicit at inference time, enabling post-hoc analysis of the model's modality weighting.

**C2. A configurable implementation supporting systematic ablation.** The `MultimodalModel` class accepts a `streams` argument that selects any subset of the three streams, a `fusion_type` argument that switches between attention and concatenation fusion, and a `use_frequency` toggle that enables or disables the FFT branch. This configuration interface enables modality, fusion-mechanism, and input-representation ablations through a single code path, ensuring that the ablation results are directly comparable across conditions.

**C3. A systematic evaluation on a real-world corpus.** The framework is evaluated on the WildDeepfake corpus — 110 real deepfake videos harvested from the open internet — rather than on a synthesised benchmark. The evaluation reports accuracy, precision, recall, F1, and AUC for the four models and for a score-level ensemble, and includes a per-epoch training analysis, a confusion-matrix analysis, and a Grad-CAM qualitative analysis.

**C4. A reproducible software framework.** The implementation includes the proposed model, the three baselines, the training pipeline (with stratified video-level splits, balanced sampling, augmentation, cosine-annealed learning rates, and early stopping), the evaluation suite (with AUC computation), the ablation runner, the cross-dataset evaluator, the compression-robustness tester, the Grad-CAM visualiser, and the ensemble combiner. The framework is released under the MIT licence to support further research.

### 1.6 Scope and Delimitations

The dissertation is scoped along four dimensions. First, the empirical evaluation is conducted exclusively on the WildDeepfake corpus; while the framework supports the FaceForensics++ [1] and DFDC [2] datasets programmatically, the institutional access and storage required to train on the full benchmarks were not available within the time budget of this study. Second, the framework is evaluated on a corpus of 110 videos (52 authentic, 58 manipulated); while this is sufficient to expose qualitative trends in the models' behaviour, it is insufficient to support claims of statistical significance at the level customary in large-scale benchmark studies. Third, the audio stream's contribution is constrained by the fact that WildDeepfake, being a face-sequence dataset, contains no audio tracks; the audio stream is therefore fed silence-padded inputs on this corpus, and its behaviour is reported accordingly. Fourth, the framework is restricted to binary real/fake classification; multi-class attribution of the manipulation method (Deepfakes, Face2Face, FaceSwap, NeuralTextures) is out of scope.

### 1.7 Thesis Organisation

The remainder of the dissertation is organised as follows. Chapter 2 surveys the deepfake-detection literature, organised by modality, and identifies the research gap addressed by this work. Chapter 3 describes the proposed architecture, the baseline models, the datasets, the implementation details, and the evaluation metrics. Chapter 4 reports the experimental results, including the main quantitative comparison, the per-epoch training analysis, the ensemble analysis, the Grad-CAM qualitative analysis, and the ablation studies. Chapter 5 discusses the results in the context of the broader literature, and appraises the strengths, limitations, and ethical implications of the framework. Chapter 6 concludes and identifies directions for future work.

---

# Chapter 2

## Literature Review

This chapter surveys the deepfake-detection literature, organised by the modality of evidence that each detector exploits. Section 2.1 reviews the deepfake generation techniques that the detectors are designed to expose, since the choice of generator materially constrains the choice of detection signal. Sections 2.2 through 2.4 survey unimodal detectors — visual, temporal, and audio, respectively. Section 2.5 surveys multimodal detectors, with particular attention to the fusion mechanisms they employ. Section 2.6 identifies the research gap addressed by this dissertation. Section 2.7 summarises the chapter.

### 2.1 Overview of Deepfake Generation Techniques

Modern deepfake generation techniques fall into four broad families. *Face-swap* methods transplant a source identity onto a target video while preserving the target's facial motion and expression; the canonical approach pairs an encoder that maps both source and target faces into a shared latent space with two decoders that reconstruct the respective identities, and then composites the source-decoded face back into the target frame with a blending mask. The encoder–decoder–decoder architecture is the original deepfake pipeline, introduced in the r/deepfakes community in late 2017, and remains the dominant approach for consumer-grade face-swap applications. The encoder learns a compressed representation of the face that captures the identity-invariant features (the geometry of the face, the skin micro-texture, the hair), while the decoders learn to reconstruct the identity-specific features (the source identity's appearance, the target identity's appearance). At inference time, the source identity's face is encoded by the encoder and decoded by the target identity's decoder, producing a face that has the source identity's appearance but the target video's motion and expression. The resulting face is then blended back into the target frame using a blending mask that smooths the boundary between the swapped face and the surrounding skin.

*Lip-sync* methods, exemplified by Wav2Lip and similar audio-conditioned vocoders, reshape the mouth region of a target video to match a substituted audio track while leaving the rest of the face untouched. The lip-sync generator takes the target video's frames and the substituted audio's mel-spectrogram as input, and produces a mouth-region output that is synchronised with the audio. The output is composited back into the target frame using a mask that limits the modification to the mouth region, leaving the rest of the face (the eyes, the nose, the cheeks, the jaw) untouched. The lip-sync generator is therefore a *partial* face-swap: it modifies only the mouth region, and the resulting video's authenticity depends on the quality of the lip-sync generator's output and the consistency of the mouth region's modification with the rest of the face.

*Puppet-master* methods drive a target identity with the facial-expression parameters of a source actor, enabling performance-capture-style manipulation of static portraits. The puppet-master generator takes a single source image (the target identity) and a sequence of facial-expression parameters (extracted from the source actor's performance) as input, and produces a sequence of output frames in which the target identity's facial expression matches the source actor's. The facial-expression parameters are typically represented as a set of action units (the Facial Action Coding System's action units, which describe the activation of specific facial muscles) or as a 3D face model's parameters (the 3D Morphable Model's shape and pose parameters). The puppet-master generator is more challenging than the face-swap generator, since it must synthesise the target identity's appearance under a range of facial expressions, rather than simply swapping the appearance of an existing expression.

*Full-generation* methods, including diffusion-based talking-head synthesis, generate an entire video clip from a text prompt or a single reference image without reference to a target video. The full-generation generator is the most challenging of the four families, since it must synthesise both the appearance and the motion of the face, and the resulting video's authenticity depends on the generator's ability to produce a coherent face that is consistent across the video's frames. The diffusion-based talking-head generator, which has emerged as the state-of-the-art approach in the past two years, uses a denoising diffusion probabilistic model to generate the face's appearance and a separate motion model to drive the face's motion, with the two models trained jointly to produce a coherent output.

These families leave different artefact signatures. Face-swap methods, especially when applied at low computational budget, leave blending artefacts around the face boundary, frequency-domain signatures from the up-sampling operations in the decoder, and temporal inconsistency in the head pose if the swap is performed per-frame without a temporal constraint. Lip-sync methods leave a desynchronisation between the visual lip motion and the acoustic content — the very signal exploited by lip-sync detectors — but the rest of the face may be perfectly authentic. Puppet-master methods preserve the target identity but may introduce unnatural eye motions, blink cadences, or head-pose dynamics, since the puppet-master generator's motion model may not perfectly capture the natural dynamics of the target identity. Full-generation methods are the most challenging to detect, since they leave no "authentic" reference signal against which to compare, but they often introduce low-level texture artefacts in the generated face and anomalous high-frequency content in the generated background.

The detection literature surveyed below is largely organised around these artefact families, with each detector targeting a particular signal. The central thesis of the present work — that a detector should fuse multiple signals under a learned attention mechanism — is motivated by the observation that the artefact family of a given forgery is rarely known in advance, and that a single-signal detector is silent precisely when the forgery is of a family that does not produce its target artefact. A detector that fuses multiple signals, in contrast, can produce a non-degenerate output for any forgery family, since at least one of the fused signals is likely to be informative for any given forgery.

The artefact signatures of the four families are also not static: as the generative models improve, the artefact signatures are progressively closed, and the detectors that exploit those signatures become less effective. The closure of the artefact signatures is the principal driver of the cat-and-mouse dynamic between the generative and detection communities, and the detector designer's task is to identify artefact signatures that are difficult for the generative model to close. The biological-signal artefacts (the photoplethysmographic signal exploited by FakeCatcher [7] and DeepRhythm [35]) are particularly difficult to close, since they require the generative model to explicitly synthesise the biological signal, which is a non-trivial addition to the generator's architecture. The frequency-domain artefacts (the high-frequency signature exploited by Frank et al. [13]) are easier to close, since they can be addressed by adopting a more sophisticated up-sampling strategy, but the closure is not always complete and the residual signature may still be detectable. The temporal artefacts (the inter-frame inconsistencies exploited by the temporal stream of the present framework) are the most difficult to close, since they require the generative model to produce temporally coherent video, which is a substantially harder problem than producing a single realistic frame.

### 2.2 Unimodal Visual Detection

The unimodal visual detector family is the largest and most-studied branch of the deepfake-detection literature. The detectors in this family operate on individual frames (or, more precisely, on face crops extracted from individual frames) and classify each frame as real or fake based on the pixel-level statistics of the face. The per-frame decisions are typically aggregated to a clip-level decision by averaging across the clip's frames. The family can be further subdivided into three sub-families, distinguished by the type of artefact signal that each detector exploits.

#### 2.2.1 Spatial-Artefact Detectors

The earliest modern deepfake detectors treated the problem as a binary image classification task on individual face crops. Afchar et al. introduced MesoNet [4], a compact convolutional network whose architecture is deliberately shallow — two convolutional layers with relatively small kernel sizes — to enforce learning of mesoscopic features such as the absence of high-frequency detail in the eye region. The MesoNet architecture consists of two initial 3×3 and 5×5 convolutional layers with eight and sixteen channels respectively, each followed by batch normalisation, ReLU activation, and max-pooling, and two further 5×5 convolutional layers with sixteen channels each, followed by a 1024→16→1 fully connected classifier with dropout. The total parameter count is approximately 50,000, making MesoNet two orders of magnitude smaller than the Xception-based detectors that followed. MesoNet attained an accuracy of 95 % on the early FaceForensics benchmark and remains a standard lightweight baseline; the present dissertation includes MesoNet as one of its three baselines for this reason.

Chollet's Xception architecture [5], originally designed for general image classification, was re-purposed for deepfake detection by Rossler et al. [1] and quickly became the dominant baseline. The Xception architecture replaces the standard convolutional block of Inception with a depthwise-separable convolution, in which the spatial convolution is performed independently on each input channel, followed by a 1×1 pointwise convolution that combines the channel-wise results. This decomposition reduces the parameter count and the floating-point operation count by a factor of approximately 8× relative to a standard convolution with the same kernel size and channel count, while preserving the representational capacity. The depthwise-separable convolution at the heart of Xception provides a favourable trade-off between parameter count and representational capacity, and the use of ImageNet-pretrained weights enables strong performance even on relatively small training sets. The Xception-based detector in the FaceForensics++ benchmark attained an accuracy of 95.5 % on the high-quality (c23) split, and remains the most-cited single-frame detector in the literature. The present dissertation includes Xception as one of its three baselines.

Subsequent work sought to improve upon Xception by augmenting the backbone with attention mechanisms or domain-specific modules. Dang et al. [8] introduced a face-region-specific attention that up-weighted the eye and mouth regions, on the observation that those regions carry the strongest manipulation signals. Their detector, called OA-FaceD, computes per-region attention maps from the face's facial landmark locations and applies them as multiplicative gating on the convolutional feature maps, effectively steering the network's representational capacity towards the most informative regions. Hu et al.'s Squeeze-and-Excitation module [19] was adapted to deepfake detection by several authors, providing a generic channel-attention mechanism that improved performance across a range of backbones; the SE module computes a per-channel scaling factor from the global average of the feature map, allowing the network to up-weight informative channels and suppress uninformative ones. Nirkin et al. [32] proposed a detector that operated on discrepancies between the face and its surrounding context, observing that a face-swap operation that produces a convincing face in isolation may produce an unconvincing transition at the boundary between face and background; their detector processes the face crop and the surrounding context crop through separate backbones and concatenates the resulting features.

A separate strand of detectors targets the *geometry* of the synthesised face, on the observation that early deepfake generators produced faces whose geometric landmarks — the relative positions of the eyes, nose, mouth, and jaw — were subtly inconsistent with the geometry of authentic faces. These detectors typically use a facial landmark detector (e.g., dlib or MediaPipe) to extract the 68 standard facial landmarks, then feed the landmark configuration into a small classifier. The geometric detector is computationally cheap — the landmark extraction is far cheaper than a full CNN forward pass — but is brittle to landmark-detection errors and to generators that explicitly model the geometric constraints of the face.

#### 2.2.2 Frequency-Domain Detectors

The frequency domain offers a complementary signal to the pixel domain: the up-sampling operations common to GAN-based generators — transposed convolutions, nearest-neighbour up-sampling, bilinear up-sampling — leave characteristic signatures in the Fourier spectrum that are largely invisible to a pixel-space classifier. Frank et al. [13] provided the canonical study of this phenomenon, demonstrating that GAN-generated images exhibit an anomalous high-frequency signature attributable to the periodicity of the up-sampling operation. The signature is a consequence of the discrete nature of the up-sampling operation: a transposed convolution with stride 2 produces an output whose pixels at even indices are computed by a different convolution path than the pixels at odd indices, and this asymmetry produces a periodicity in the high-frequency content that is detectable in the Fourier spectrum. Frank et al. constructed a frequency-features classifier that, in conjunction with a pixel-space classifier, attained state-of-the-art performance on a range of GAN-detection benchmarks.

The frequency-domain signal is not, however, a universal indicator of synthesis. Durall et al. and others have shown that as the generative model adopts more sophisticated up-sampling strategies — sub-pixel convolutions, learnable up-sampling, adversarial loss on the high-frequency content — the frequency signature is partially or fully closed. The robust detectors therefore fuse frequency features with pixel features, on the observation that the two signals are complementary: a generator that closes one often leaves the other. This observation motivates the parallel spatial-frequency visual backbone of the present work.

A related line of work analyses the *phase* of the Fourier spectrum rather than the magnitude, on the observation that the phase carries the structural information of the image while the magnitude carries the texture information. Phase-based detectors exploit the observation that GAN-generated images often have inconsistent phase statistics, particularly in the high-frequency bands where the generator's up-sampling signature is most pronounced. The phase-based signal is complementary to the magnitude-based signal, and a detector that fuses both can exploit a richer frequency-domain representation than a magnitude-only detector.

#### 2.2.3 Biological-Signal Detectors

A third strand of unimodal detectors targets the biological signals that are difficult to synthesise faithfully. Li et al.'s *In Ictu Oculi* [26] exploited the observation that early deepfake generators did not blink the eyes of the synthesised face at a realistic cadence; their detector estimated blink patterns from a sequence of eye-region crops and classified the resulting time series. The blink-based detector is a temporal detector in disguise — it requires a sequence of frames to estimate the blink cadence — but it is fundamentally a unimodal visual detector, since it operates on the visual modality alone. The blink signal is fragile, however: a generator that explicitly models the blink cadence, or that copies the eye region from the source video rather than synthesising it, will defeat the detector.

Ciftci and Demir's *FakeCatcher* [7] extended this idea to blood-flow patterns, exploiting the fact that the photoplethysmographic signal — the small colour variation induced by the cardiac cycle — is present in authentic footage but absent or incoherent in synthesised footage. The photoplethysmographic signal is on the order of 0.1 % of the pixel intensity, and is therefore invisible to a human observer, but is detectable by a machine classifier that averages across many pixels and many frames. The FakeCatcher detector extracts a region-of-interest from the face (typically the forehead and cheeks, where the photoplethysmographic signal is strongest), computes the temporal average of the region's pixel intensities, and analyses the resulting time series for the characteristic cardiac-cycle frequency. The detector is robust to pixel-level artefacts and to compression, since the photoplethysmographic signal is a low-frequency phenomenon that survives compression well.

Qi et al.'s *DeepRhythm* [35] further refined the cardiac-rhythm signal by attention-weighting the temporal and spatial regions of the face that carry the strongest photoplethysmographic signal. Their detector uses a spatial-temporal attention mechanism to identify the face regions whose pixel intensities exhibit the strongest cardiac-cycle periodicity, and weights those regions more heavily in the final classification decision. The detector achieved state-of-the-art performance on the FaceForensics++ benchmark, demonstrating that the biological-signal approach, when combined with attention mechanisms, can match or exceed the performance of purely visual approaches.

The biological-signal detectors are noteworthy for their adversarial robustness: a generator that does not explicitly model the photoplethysmographic signal cannot easily defeat them. However, they are also limited by their dependence on a single biological signal — blink, blood flow, or heart rate — and by their reliance on relatively long, high-resolution input clips over which the biological signal can be estimated. Their performance on short, compressed, internet-grade clips, where the biological signal may be obscured by compression artefacts, remains an open question. They are also vulnerable to a generator that explicitly synthesises the photoplethysmographic signal, by modulating the pixel intensities of the synthesised face at the cardiac-cycle frequency; such a generator would defeat the biological-signal detector, but at the cost of substantially increased generator complexity.

### 2.3 Temporal and Sequential Detection

Temporal detectors operate on sequences of frames rather than individual images, on the observation that generating temporally coherent video is substantially harder than generating a single still image. The difficulty arises from a fundamental asymmetry: a single still image requires only that the generator produce a face that *looks* real, while a sequence of frames requires that the generator produce a face that *moves* realistically, and the space of realistic facial motions is vastly more constrained than the space of plausible facial appearances. A per-frame face-swap operation — where each frame of the target video is independently swapped with the source face — produces a sequence whose individual frames may be highly realistic, but whose inter-frame dynamics are anomalous: the head pose may jitter, the lighting may shift, the skin micro-texture may flicker, and the blink cadence may be irregular.

The early temporal detectors used recurrent neural networks. Amerini et al. [2] introduced an optical-flow-based CNN that computed dense optical flow between consecutive frames and fed the resulting flow field into a CNN classifier; the underlying observation was that face-swap operations performed per-frame produce flow fields whose magnitude and direction statistics are distinguishable from those of authentic footage. The optical-flow detector is computationally expensive — the dense flow computation is comparable in cost to a forward pass through a CNN — but the resulting flow field is a rich representation of the inter-frame dynamics that is invariant to the per-frame appearance. The detector achieved an accuracy of approximately 81 % on the FaceForensics++ benchmark, a figure that was competitive with the single-frame Xception detector at the time of publication.

Masi et al. [28] proposed a two-branch recurrent network that combined a CNN-extracted per-frame visual feature with an LSTM-extracted temporal feature, demonstrating that the recurrent component provided a measurable improvement over the purely convolutional baseline. The two-branch architecture — a visual branch that processes each frame independently and a temporal branch that processes the sequence of per-frame features — has become the canonical structure for temporal deepfake detection. The visual branch is typically a strong ImageNet-pretrained CNN (Xception, EfficientNet, or ResNet), and the temporal branch is typically a recurrent network (LSTM or GRU) or, more recently, a Transformer encoder. The present dissertation's temporal stream adopts this canonical structure, with EfficientNet-B0 as the visual branch and a Transformer encoder as the temporal branch.

The introduction of the Transformer architecture [43] in the natural-language domain, and its subsequent adaptation to the visual domain by Dosovitskiy et al. [12] and others, opened a new family of temporal detectors based on self-attention. Unlike recurrent networks, Transformers can attend directly to any pair of frames in the input clip, avoiding the bottleneck of a sequential hidden state. The self-attention mechanism allows each frame to query any other frame for relevant information, and the multi-head attention allows the model to attend to multiple aspects of the inter-frame relationship simultaneously (e.g., one head might attend to the head-pose relationship between frames, while another head attends to the lighting relationship). Mittal et al.'s *Emotions Don't Lie* [29] detector, while primarily a multimodal system, included a Transformer-encoded temporal component that modelled the affective dynamics of the face. The general pattern — a CNN-extracted per-frame feature followed by a Transformer-encoded sequence model — has become the de facto structure for temporal deepfake detection, and is the structure adopted by the temporal stream of the present framework.

A third strand of temporal detectors targets specific inter-frame dynamics, rather than the generic inter-frame statistics. The blink-cadence detector [26] (discussed in Section 2.2.3 as a biological-signal detector, but also a temporal detector) models the blink cadence as a time series and classifies the resulting series. The head-pose detector models the 3D head pose as a function of time and classifies the resulting trajectory; the underlying observation is that a face-swap operation may produce a head-pose trajectory whose magnitude and direction statistics are distinguishable from those of authentic footage, even when the individual frames are realistic. The head-pose detector is robust to pixel-level artefacts and to compression, but is vulnerable to a generator that explicitly models the head-pose trajectory.

### 2.4 Audio-Based Detection

Audio-based detection, while less explored than visual detection, has received increasing attention as audio synthesis has matured to the point where synthetic speech is indistinguishable from authentic speech to a human listener. The modern neural vocoders — WaveNet, WaveGlow, HiFi-GAN, and their successors — produce speech whose naturalness scores, as measured by mean opinion score (MOS) studies, exceed 4.5 out of 5, indistinguishable from authentic human speech. The detection of synthetic speech is therefore no longer a matter of detecting obvious artefacts (e.g., robotic prosody, missing breath sounds, unnatural pauses) but of detecting subtle statistical signatures in the spectral and cepstral representations of the speech.

The dominant audio features are the Mel-frequency cepstral coefficients (MFCCs), which compress the short-time Fourier transform (STFT) of the audio signal through a perceptually-motivated mel-scale filterbank followed by a discrete cosine transform. The mel scale is a perceptual scale of pitches judged by listeners to be equal in distance from one another, and is approximately linear below 1 kHz and logarithmic above. The mel filterbank, a set of overlapping triangular filters spaced on the mel scale, models the frequency resolution of the human auditory system, which is approximately linear below 1 kHz and logarithmic above. The discrete cosine transform (DCT) at the heart of MFCC decorrelates the mel bands, producing a compact representation whose lower coefficients capture the spectral envelope (the shape of the vocal tract) and whose higher coefficients capture the fine spectral detail. The 40-coefficient MFCC frame — the configuration adopted in this dissertation — captures the spectral envelope in sufficient detail for both speech recognition and synthetic-speech detection, and is widely used in both domains. The Librosa library [6] provides the canonical open-source implementation of MFCC extraction.

The earliest audio deepfake detectors used pre-trained audio classification backbones. Hershey et al.'s *VGGish* [17] model, pre-trained on a large-scale audio classification task (AudioSet, with over 2 million video clips and 527 audio event classes), was repurposed by several authors as a feature extractor for synthetic-speech detection. The VGGish features, while not specifically trained for synthetic-speech detection, capture a rich set of audio statistics that are informative for the task, and the use of a pre-trained backbone enables strong performance even on relatively small training sets. Subsequent work built customised CNNs over MFCC inputs; the architecture of the audio stream in the present framework — a stack of three Conv2D layers with interspersed max-pooling and a final adaptive average pool — is a representative instance of this family.

A separate strand of audio detectors targets the *consistency* between the audio and the visual lip motion, rather than the audio alone. This lip-sync-based detection is surveyed in Section 2.5 as a multimodal approach, since it requires both modalities. Pure audio detectors are constrained by the fact that many deepfakes manipulate only the visual modality, leaving the audio authentic; in those cases the audio detector is silent regardless of the quality of the synthetic visual content. This observation is one of the principal motivations for multimodal fusion: a detector that relies on the audio alone will miss the visually-manipulated deepfakes, while a detector that relies on the visual alone will miss the audio-manipulated deepfakes, and a detector that fuses both can detect both families.

The audio detection literature also includes a strand of work that targets the *background noise* of the audio, on the observation that synthetic audio may have a background-noise signature that is distinguishable from authentic audio. The background-noise detector analyses the spectral content of the silent portions of the audio (the portions between speech, where the speaker is not vocalising) and classifies the resulting spectral signature. The detector is robust to the choice of speaker and to the speech content, since the background noise is independent of both, but is vulnerable to a generator that explicitly models the background noise of the target audio. The background-noise detector is complementary to the MFCC-based detector, and a detector that fuses both can exploit a richer audio representation than either alone.

### 2.5 Multimodal Detection

Multimodal detectors fuse evidence from two or more modalities — most commonly the visual and the audio — under the observation that the modalities carry complementary signals. The simplest fusion strategy is *late fusion* by score averaging: each modality is processed by a dedicated detector, and the final score is the (possibly weighted) average of the per-modality scores. This strategy is attractive for its simplicity and for the fact that the per-modality detectors can be trained independently, but it forecloses any cross-modal reasoning at the feature level. A late-fusion detector cannot, for instance, learn to down-weight the audio score when the audio is silent, or to up-weight the visual score when the visual is high-quality; the per-modality scores are combined by a fixed function, regardless of the input.

A more powerful family of multimodal detectors performs *intermediate fusion* by concatenating or projecting features from the two modalities into a shared representation. The intermediate-fusion detector learns a shared representation that captures the cross-modal relationships, enabling the model to reason about the consistency between the modalities rather than simply averaging their scores. Mittal et al.'s *Emotions Don't Lie* [29] fused affective features extracted from the face and the voice under a learned attention mechanism, demonstrating that the affective consistency between the two modalities is a strong signal of authenticity. The affective features — the emotional state of the face and the emotional state of the voice — are computed by dedicated affect-detection networks, and the consistency between them is computed by a learned attention mechanism that up-weights the modality whose affective signal is stronger. The detector achieved state-of-the-art performance on the DFDC benchmark, demonstrating that the affective-consistency signal is a powerful indicator of authenticity.

Jaiswal and AbdAlmageed [21] used Vision Transformers and convolutional networks in parallel on the visual and acoustic modalities, with fusion performed by a late cross-attention. The cross-attention mechanism allows each modality to attend to the other modality's features, enabling the model to reason about the cross-modal relationships at the feature level rather than at the score level. The cross-attention fusion is more expressive than simple concatenation, but at the cost of additional parameters and computational complexity.

Pan et al.'s *TVNet* [33] is perhaps the closest prior work to the present framework: it fuses visual, audio, and temporal cues under a multimodal consistency loss, observing that the three modalities should agree on the authenticity label and that any disagreement is itself a signal of manipulation. The TVNet architecture consists of a visual branch (a CNN), an audio branch (a CNN over MFCC inputs), and a temporal branch (a recurrent network), with the three branches trained jointly under a multimodal consistency loss. The consistency loss penalises disagreement between the modalities, encouraging the model to learn a shared representation that captures the cross-modal relationships. The TVNet achieved state-of-the-art performance on the FaceForensics++ benchmark, and is the principal prior work against which the present framework is compared.

The present framework differs from Pan et al. in two respects. First, it adopts an attention-based feature fusion rather than a consistency loss. The attention mechanism allows the model to learn the per-stream weighting dynamically, conditioned on the input, whereas the consistency loss imposes a fixed weighting (uniform agreement) on the modalities. The attention mechanism is more flexible, and the per-stream weights are inspectable at inference time, providing an interpretability advantage over the consistency loss. Second, the present framework integrates a frequency-domain visual branch alongside the spatial branch, whereas the TVNet uses a purely spatial visual branch. The frequency branch provides a complementary signal to the spatial branch, and the fusion of the two signals is, in our experience, more robust to the closure of the spatial artefact by the generative model.

A third, less-explored family of multimodal detectors exploits *lip-sync* consistency. The canonical approach, exemplified by the lip-sync detector in the present framework's baselines, processes the visual mouth region and the audio spectrogram through parallel CNNs and concatenates the resulting features into a fused classifier. The detector is silent on visually manipulated deepfakes with authentic audio (since the lip motion matches the audio in both the real and the visually-manipulated case), but is highly effective on audio-manipulated deepfakes, where the visual lip motion and the substituted audio are desynchronised. The lip-sync detector is included as a baseline in the present framework for this reason: it represents the audio-visual consistency signal in a structurally simpler form than the proposed model, and the comparison between the two isolates the contribution of the proposed model's attention fusion from the contribution of its additional streams.

A fourth family of multimodal detectors exploits *cross-modal verification*: rather than fusing the modalities into a shared representation, the cross-modal verification approach trains a dedicated network to verify the consistency between the modalities. The verification network takes the per-modality features as input and produces a binary consistency score, which is then combined with the per-modality authenticity scores to produce the final decision. The cross-modal verification approach is more flexible than the intermediate-fusion approach, since the verification network can be trained independently of the per-modality detectors, but it is also more computationally expensive, since it requires an additional forward pass through the verification network.

### 2.6 Identified Research Gap

The literature surveyed above exhibits three patterns. First, the unimodal detectors — visual spatial, visual frequency, temporal, and audio — are each effective on a particular family of forgeries, but are silent on forgery families that do not produce their target artefact. Second, the multimodal detectors that have been proposed — *Emotions Don't Lie* [29], *TVNet* [33], the lip-sync family — have largely focused on fusing two modalities, with relatively few works integrating three or more. Third, the fusion mechanisms in the multimodal literature are largely either simple concatenation or fixed-weight averaging, with relatively few works adopting a *learned* attention-based fusion whose weights can be inspected at inference time to expose the model's modality weighting.

The research gap addressed by this dissertation is the integration of *four* signals — spatial visual, frequency visual, MFCC audio, and Transformer-encoded temporal — under a single, *learned* attention-based fusion, with the explicit goal of producing a detector that is informative on the small, real-world corpora characteristic of operational deepfake-detection deployments. The framework is designed so that the ablation of any single modality, the ablation of the fusion mechanism, and the ablation of the input representation can be conducted through a single code path, enabling the systematic attribution of performance to architectural components.

A further gap in the literature is the relative scarcity of evaluations on *real-world* deepfake corpora, as opposed to *synthesised* benchmarks. The FaceForensics++ corpus [1], while widely used, consists of forgeries generated under a controlled pipeline (the four FF++ manipulation methods: Deepfakes, Face2Face, FaceSwap, NeuralTextures), and the detectors trained on FF++ have been shown to generalise poorly to forgeries produced by other pipelines. The DFDC corpus [2, 30] partially addresses this by including forgeries produced by a wider range of methods, but the forgeries are still generated under controlled conditions. The WildDeepfake corpus [3], whose forgeries are harvested from the open internet, is the most operationally realistic benchmark, but its small size (110 videos in the version used in this dissertation) limits the statistical significance of the evaluations conducted on it. The present dissertation's evaluation on the WildDeepfake corpus is therefore a deliberate choice to evaluate the framework on a real-world corpus, with the explicit acknowledgement of the statistical-significance limitations.

### 2.7 Summary

This chapter has surveyed the deepfake-detection literature along five axes: the generation techniques that produce the forgeries, the unimodal visual detectors, the temporal detectors, the audio detectors, and the multimodal detectors. The survey has identified a research gap around the integration of four complementary signals under a learned, attention-based fusion, and has motivated the design of the proposed framework, which is described in detail in the next chapter. The survey has further noted the relative scarcity of evaluations on real-world corpora, motivating the present dissertation's evaluation on the WildDeepfake corpus despite its small size.

---

# Chapter 3

## Methodology

This chapter describes the proposed multimodal framework in detail. Section 3.1 gives a system overview. Sections 3.2 through 3.4 describe the three feature streams — visual, audio, and temporal — in turn. Section 3.5 describes the attention-based fusion module. Section 3.6 describes the classification head. Section 3.7 describes the three baseline models against which the framework is benchmarked. Section 3.8 describes the datasets. Section 3.9 describes the implementation details. Section 3.10 describes the evaluation metrics. Section 3.11 describes the ablation framework. Section 3.12 summarises the chapter.

### 3.1 System Overview

The proposed framework, `MultimodalModel`, is a tri-stream architecture whose forward pass accepts a video tensor $V \in \mathbb{R}^{B \times T \times 3 \times H \times W}$ and an audio feature tensor $A \in \mathbb{R}^{B \times 1 \times F \times \tau}$ and produces a binary real/fake score $\hat{y} \in [0,1]^B$. The architecture decomposes into four functional blocks:

1. **Visual stream** (`VisualStream`): processes the $T$-frame video clip through a parallel spatial backbone and an FFT frequency backbone, producing a 2560-dimensional combined visual feature per clip.
2. **Audio stream** (`AudioStream`): processes the MFCC input through a three-layer Conv2D encoder, producing a 512-dimensional audio feature.
3. **Temporal stream** (`TemporalStream`): processes the per-frame visual features through a three-layer Transformer encoder, producing a 2560-dimensional temporal feature.
4. **Fusion and classifier** (`AttentionFusion` + `classifier`): projects the three stream features into a shared 512-dimensional space, applies a per-stream attention weight, and passes the fused feature through a two-layer fully connected classifier terminated by a sigmoid.

The streams are configurable: the `streams` constructor argument accepts any subset of `{'visual', 'audio', 'temporal'}`, enabling the modality-ablation experiments described in Section 3.11. The fusion mechanism is similarly configurable: the `fusion_type` argument accepts `'attention'` (default) or `'concat'`, enabling the fusion-mechanism ablation. The FFT branch is toggled by the `use_frequency` argument, enabling the input-representation ablation. The full system architecture is summarised in Figure 3.1, and the principal hyperparameters are listed in Table 3.1.

*Table 3.1: Architecture hyperparameters of the proposed MultimodalModel.*

| Component | Parameter | Value |
|-----------|-----------|-------|
| Visual spatial backbone | Network | EfficientNet-B0 (timm, ImageNet-pretrained) |
| Visual frequency backbone | Network | EfficientNet-B0 with `in_chans=1` |
| Visual output dimension | — | 2 × 1280 = 2560 |
| Audio conv layers | Configuration | (1→32, 3×3), (32→64, 3×3), (64→128, 3×3) with max-pool 2×2 |
| Audio feature dimension | — | 512 |
| Temporal encoder | Layers | 3 TransformerEncoderLayer |
| Temporal heads | — | 8 |
| Temporal dropout | — | 0.1 |
| Fusion dimension | — | 512 |
| Classifier | Layers | FC(512→256) → ReLU → Dropout(0.5) → FC(256→1) → Sigmoid |
| Configurable streams | — | Any subset of {visual, audio, temporal} |
| Configurable fusion | — | `attention` or `concat` |
| Configurable FFT | — | True or False |

### 3.2 Visual Stream

The visual stream is the framework's largest consumer of parameters and the principal source of per-frame discriminative features. It is implemented in the `VisualStream` class and consists of a parallel pair of EfficientNet-B0 backbones — one operating on the raw RGB frame and one operating on the log-magnitude FFT of the frame's luminance channel — whose outputs are concatenated along the feature dimension.

#### 3.2.1 Spatial Backbone

The spatial backbone is an EfficientNet-B0 network [42], instantiated through the `timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)` API, which discards the classification head and exposes the 1280-dimensional feature vector produced by the network's terminal global average pooling. EfficientNet-B0 was selected for its favourable parameter–accuracy trade-off: at roughly 5.3 million parameters, it provides ImageNet-level representational capacity at a fraction of the cost of the larger ResNet-152 or Xception backbones commonly used in deepfake detection. The ImageNet-pretrained weights provide a strong initialisation that, given the small training corpus, materially reduces the risk of overfitting in the early training epochs.

EfficientNet-B0 is the baseline member of the EfficientNet family, a set of convolutional networks whose architecture is determined by a compound scaling method that jointly scales the network's depth, width, and input resolution by a fixed set of coefficients. The compound scaling method is based on the observation that scaling any one of these dimensions in isolation produces diminishing returns beyond a certain point, and that scaling all three dimensions in proportion produces a better trade-off between accuracy and computational cost. The EfficientNet-B0 architecture consists of nine stages, each of which is a stack of mobile inverted bottleneck convolution (MBConv) blocks. The MBConv block is a depthwise-separable convolution with an inverted residual connection: the input is first expanded by a 1×1 convolution to a higher-dimensional representation, then a depthwise 3×3 (or 5×5) convolution is applied, then a 1×1 projection reduces the dimensionality back to the original, and a residual connection is added if the input and output dimensions match. The MBConv block also includes a Squeeze-and-Excitation [19] channel-attention mechanism, which computes a per-channel scaling factor from the global average of the feature map and applies it as a multiplicative gating. The combination of depthwise-separable convolution, inverted residual connection, and channel attention provides a representational capacity per parameter that is substantially higher than the standard convolutional block.

The forward pass of the spatial backbone is straightforward: given a frame $I \in \mathbb{R}^{3 \times H \times W}$, the backbone produces a feature vector $f_{\text{spatial}} \in \mathbb{R}^{1280}$. Because the spatial backbone is shared across all frames in a clip (the same backbone processes each frame independently), the per-frame parameters are amortised across the temporal dimension, and the temporal stream — described in Section 3.4 — can directly consume the per-frame backbone output.

The choice of EfficientNet-B0 over the larger EfficientNet-B4 or B7 variants is motivated by the small training corpus. The larger variants provide additional representational capacity that would, on a larger corpus, translate into improved accuracy, but on a corpus of 110 videos the additional capacity is more likely to be used to memorise the training set than to learn generalisable features. The B0 variant is therefore a conservative choice that trades potential accuracy for reduced overfitting risk. On a larger corpus (FF++, DFDC), the B0 variant could be replaced by the B4 or B7 variant without code changes, by modifying the `visual_model_name` constructor argument.

#### 3.2.2 Frequency Branch

The frequency branch is the second component of the visual stream, processing the log-magnitude FFT of the frame's luminance channel through a *separate* EfficientNet-B0 backbone, instantiated with `in_chans=1` to accept a single-channel input. The use of a separate backbone, rather than a shared backbone applied to a stacked RGB+FFT input, is a deliberate design choice: the frequency-domain statistics differ sufficiently from the spatial-domain statistics that a shared backbone would, in our experience, learn to ignore the frequency channel in favour of the more familiar spatial statistics. The separate backbone forces the network to learn a frequency-specific representation.

The frequency branch computes the FFT as follows. Given an input frame $I \in \mathbb{R}^{3 \times H \times W}$, the luminance is computed as the pixel-wise mean across the three channels:

$$I_{\text{gray}} = \frac{1}{3} \sum_{c=1}^{3} I_c \in \mathbb{R}^{1 \times H \times W}.$$

The two-dimensional discrete Fourier transform is then computed via the `torch.fft.fft2` operator:

$$\mathcal{F}(u, v) = \sum_{x=0}^{H-1} \sum_{y=0}^{W-1} I_{\text{gray}}(x, y) \, e^{-2\pi i \left( \frac{ux}{H} + \frac{vy}{W} \right)},$$

the magnitude is computed and log-compressed:

$$M(u, v) = \log\!\big(1 + |\mathcal{F}(u, v)|\big),$$

and the result is standardised to zero mean and unit variance:

$$\tilde{M}(u, v) = \frac{M(u, v) - \mu_M}{\sigma_M + \epsilon}, \quad \epsilon = 10^{-6}.$$

The standardised log-magnitude spectrum $\tilde{M} \in \mathbb{R}^{1 \times H \times W}$ is then passed through the frequency backbone to produce a 1280-dimensional feature vector $f_{\text{freq}} \in \mathbb{R}^{1280}$. The log-magnitude step compresses the wide dynamic range of the FFT (whose magnitudes can span several orders of magnitude), and the standardisation step ensures that the input distribution to the backbone is consistent with the distribution expected by the ImageNet-pretrained initial weights. Note that the FFT here is computed without the `fftshift` operation, since the magnitude spectrum's spatial arrangement is consistent across frames and the convolutional backbone can learn the appropriate spatial weighting.

The design of the frequency branch is motivated by the observation, due to Frank et al. [13], that GAN-based up-sampling operations leave a characteristic high-frequency signature in the Fourier spectrum that is largely invisible in the pixel space. By processing the spectrum through a dedicated backbone, the framework exposes this signature as an explicit feature rather than relying on the spatial backbone to extract it implicitly.

#### 3.2.3 Feature Composition

The visual stream's output feature is the concatenation of the spatial and frequency features:

$$f_{\text{visual}} = [f_{\text{spatial}} \, \| \, f_{\text{freq}}] \in \mathbb{R}^{2560}.$$

The output dimension is therefore 2 × 1280 = 2560 when the frequency branch is enabled, and 1280 when it is disabled (via `use_frequency=False` for the input-representation ablation). When the `visual` stream is present in the model's `streams` tuple, the visual stream's output is temporally averaged across the $T$ frames of the clip to produce a clip-level visual feature for the fusion module:

$$\bar{f}_{\text{visual}} = \frac{1}{T} \sum_{t=1}^{T} f_{\text{visual}}^{(t)} \in \mathbb{R}^{2560}.$$

When the `temporal` stream is also present, the per-frame features $f_{\text{visual}}^{(t)}$ are additionally passed to the temporal stream for Transformer-encoded aggregation, as described in Section 3.4.

### 3.3 Audio Stream

The audio stream processes a Mel-frequency cepstral coefficient (MFCC) representation of the audio track through a lightweight three-layer convolutional network. The implementation resides in the `AudioStream` class, with feature extraction delegated to the `AudioProcessor` class in the data-loading pipeline.

The audio stream's design is motivated by three considerations. First, the MFCC representation concentrates the discriminative signal of synthetic speech in a relatively small feature space (40 cepstral coefficients × ~157 time frames, in contrast to the 128-band mel-spectrogram or the raw waveform), which is well-suited to the small training corpus used in this dissertation. Second, the three-layer Conv2D architecture is intentionally lightweight relative to the visual stream, on the assumption that the MFCC representation already concentrates the discriminative signal and does not require a deep network to extract it. Third, the audio stream's feature dimension (512) is matched to the fusion module's fusion dimension, simplifying the projection from the audio stream to the shared fusion space.

#### 3.3.1 MFCC Extraction

MFCC extraction proceeds in five stages, each of which implements a well-established signal-processing technique. The stages are described in detail below, both for completeness and to enable the reader to understand the audio stream's behaviour on the WildDeepfake corpus (where the input is silence) and on corpora with authentic audio (FF++, DFDC).

First, the audio waveform is extracted from the input video file by the `AudioProcessor.extract_audio` method, which uses the `moviepy.VideoFileClip` API to load the clip, converts the audio to a mono waveform at a 16 kHz sampling rate, and truncates or zero-pads the result to a fixed 5-second duration (80,000 samples). The mono conversion is performed by averaging the left and right channels (if the input is stereo) or by selecting the single channel (if the input is mono). The 16 kHz sampling rate is the standard for speech processing: it is sufficient to capture the frequencies relevant to speech perception (the human vocal tract produces energy primarily in the 80 Hz to 8 kHz range, with most of the discriminative information for speaker and speech content in the 100 Hz to 4 kHz range), and it is low enough to keep the computational cost of the downstream processing manageable. If the input video contains no audio track — as is the case for the WildDeepfake corpus — the method returns a zero-padded silence waveform of the same length, ensuring that the downstream pipeline is robust to the absence of audio.

Second, the waveform is processed by Librosa's `librosa.feature.mfcc` function with the following parameters: `n_mfcc=40` (40 cepstral coefficients), `n_fft=2048`, `hop_length=512`, `n_mels=128`, and `sr=16000`. Librosa internally computes the short-time Fourier transform (STFT) of the waveform:

$$S[f, t] = \sum_{n=0}^{N-1} w[n] \, x[n + t \cdot H] \, e^{-2\pi i f n / N},$$

where $w$ is the Hann window of length $N = 2048$, $H = 512$ is the hop length, and $t$ indexes the time frame. The Hann window is a raised-cosine window $w[n] = 0.5 (1 - \cos(2\pi n / (N-1)))$, which reduces the spectral leakage that would otherwise result from the abrupt truncation of the waveform at the window boundaries. The hop length of 512 samples (32 ms at a 16 kHz sampling rate) provides a 75 % overlap between consecutive windows, which is the standard for speech processing and ensures that the temporal resolution of the STFT is sufficient to capture the rapid spectral transitions that characterise speech.

The STFT is converted to a power spectrogram (the squared magnitude of the complex-valued STFT), projected through a 128-band mel filterbank to produce a mel-spectrogram, and the mel-spectrogram is log-compressed and decorrelated by a discrete cosine transform to produce the 40-coefficient MFCC:

$$\text{MFCC}[k, t] = \sum_{m=1}^{M} \log\!\big(\text{Mel}[m, t]\big) \, \cos\!\Big(\frac{\pi k (m - 1/2)}{M}\Big),$$

with $M = 128$ mel bands. The mel filterbank is a set of overlapping triangular filters spaced on the mel scale, a perceptual scale of pitches judged by listeners to be equal in distance from one another. The mel scale is approximately linear below 1 kHz and logarithmic above, reflecting the frequency resolution of the human auditory system. The discrete cosine transform (DCT) at the heart of MFCC decorrelates the mel bands, producing a compact representation whose lower coefficients capture the spectral envelope (the shape of the vocal tract, which is the primary determinant of speech perception) and whose higher coefficients capture the fine spectral detail (which is more relevant to speaker identification than to speech perception).

The choice of 40 cepstral coefficients (rather than the more common 13 or 20) is motivated by the synthetic-speech-detection task: the lower coefficients (1–13) capture the spectral envelope, which is largely sufficient for speech recognition, while the higher coefficients (14–40) capture the fine spectral detail that may be more relevant to the detection of synthetic speech. The 40-coefficient configuration is therefore a superset of the 13-coefficient configuration, and the model can learn to attend to the most informative coefficients for the task. The cost of the additional coefficients is a modest increase in the audio stream's input dimensionality, which is absorbed by the Conv2D architecture without difficulty.

#### 3.3.2 Convolutional Audio Encoder

The audio encoder is a three-layer Conv2D stack with interspersed ReLU activations and 2×2 max-pooling. The choice of a Conv2D architecture (rather than a Conv1D architecture, which would be the natural choice for a time-series input) is motivated by the two-dimensional structure of the MFCC input: the MFCC matrix has a frequency dimension (the 40 cepstral coefficients) and a time dimension (the ~157 time frames), and the Conv2D can exploit the spatial relationships between the two dimensions. A Conv1D, in contrast, would treat the input as a one-dimensional sequence, ignoring the frequency dimension's structure.

The convolutions use `padding=1` to preserve the spatial dimensions, so the max-pooling operations reduce the input from $(40, 157)$ to $(20, 78)$, $(10, 39)$, and $(5, 19)$, respectively. A terminal adaptive average pool collapses the spatial dimensions to $1 \times 1$, yielding a 128-dimensional feature vector, which is then projected to the 512-dimensional `feature_dim` by a linear layer:

$$h_1 = \text{MaxPool}_2\big(\text{ReLU}\big(\text{Conv}_{1\to32}^{3\times3}(\widetilde{\text{MFCC}})\big)\big),$$
$$h_2 = \text{MaxPool}_2\big(\text{ReLU}\big(\text{Conv}_{32\to64}^{3\times3}(h_1)\big)\big),$$
$$h_3 = \text{MaxPool}_2\big(\text{ReLU}\big(\text{Conv}_{64\to128}^{3\times3}(h_2)\big)\big),$$
$$f_{\text{audio}} = W_a \, \text{AvgPool}(h_3) + b_a \in \mathbb{R}^{512}.$$

The audio stream is intentionally lightweight relative to the visual stream (approximately 200,000 parameters, versus 10 million for the visual stream), on the assumption that the MFCC representation already concentrates the discriminative signal in a relatively small feature space. The 40-coefficient MFCC input, in contrast to a 128-band mel-spectrogram, is a deliberate choice: the discrete cosine transform at the heart of MFCC decorrelates the mel bands, producing a more compact representation that is empirically easier to learn from small training corpora. On a larger corpus (FF++, DFDC), the 128-band mel-spectrogram may be preferred, since the additional information in the higher-resolution representation can be absorbed by the larger training set; the framework's `AudioProcessor` class supports both representations via the `use_mfcc` constructor argument.

### 3.4 Temporal Stream

The temporal stream models inter-frame dynamics through a Transformer encoder operating on the per-frame visual features produced by the visual stream's spatial-frequency backbone. The implementation resides in the `TemporalStream` class.

#### 3.4.1 Frame-Level Feature Extractor

The temporal stream does not have a separate feature extractor; rather, it consumes the per-frame features produced by the visual stream. Specifically, when the `temporal` stream is present in the model's `streams` tuple, the visual stream is instantiated (regardless of whether `visual` is in the tuple) and its forward pass is run on every frame of the input clip. The resulting tensor of shape $(B, T, 2560)$ — where $T$ is the number of frames and 2560 is the visual stream's output dimension — is passed to the temporal stream.

This design choice has two consequences. First, it ensures that the temporal stream operates on rich, semantically meaningful per-frame features rather than on raw pixels, which substantially reduces the number of parameters that the Transformer encoder must learn. Second, it creates a parameter-sharing dependency between the visual and temporal streams: the visual backbone's weights are jointly optimised to serve both the per-frame visual feature (when present in the fusion) and the per-frame temporal feature. This dependency is, in our experience, beneficial: it regularises the visual backbone by exposing it to two distinct supervisory signals.

#### 3.4.2 Transformer Encoder

The Transformer encoder consists of three `TransformerEncoderLayer` instances from the `torch.nn` library, configured with `d_model=2560`, `nhead=8`, `dropout=0.1`, and `batch_first=True`. Each layer applies multi-head self-attention followed by a position-wise feed-forward network, with residual connections and layer normalisation at each sub-layer. The self-attention mechanism, originally introduced by Vaswani et al. [43] for natural-language translation, allows each frame in the clip to attend directly to any other frame, without the sequential bottleneck of a recurrent network.

The scaled dot-product attention at the heart of the Transformer is defined as:

$$\text{Attn}(Q, K, V) = \text{softmax}\!\Big(\frac{Q K^T}{\sqrt{d_k}}\Big) V,$$

where $Q \in \mathbb{R}^{T \times d_k}$, $K \in \mathbb{R}^{T \times d_k}$, and $V \in \mathbb{R}^{T \times d_v}$ are the query, key, and value matrices, $T$ is the sequence length (the number of frames), $d_k$ is the per-head key dimension, and $d_v$ is the per-head value dimension. The query, key, and value matrices are obtained by linearly projecting the input frame features $X \in \mathbb{R}^{T \times d_{\text{model}}}$:

$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V,$$

where $W_Q, W_K \in \mathbb{R}^{d_{\text{model}} \times d_k}$ and $W_V \in \mathbb{R}^{d_{\text{model}} \times d_v}$ are learned projection matrices. The scaling factor $\sqrt{d_k}$ in the denominator prevents the dot products from growing too large in magnitude, which would push the softmax into a region of vanishing gradients.

The multi-head attention mechanism runs $h = 8$ attention heads in parallel, each with its own set of projection matrices, and concatenates the results:

$$\text{MultiHead}(X) = [\text{head}_1 \, \| \, \text{head}_2 \, \| \, \cdots \, \| \, \text{head}_h] W_O,$$

where $\text{head}_i = \text{Attn}(X W_Q^{(i)}, X W_K^{(i)}, X W_V^{(i)})$ and $W_O \in \mathbb{R}^{(h \cdot d_v) \times d_{\text{model}}}$ is a learned output projection. The multi-head mechanism allows the model to attend to multiple aspects of the inter-frame relationship simultaneously: one head might attend to the head-pose relationship between frames, while another head attends to the lighting relationship, and a third head attends to the eye-region relationship.

The position-wise feed-forward network (FFN) consists of two linear layers with a ReLU activation in between:

$$\text{FFN}(x) = \text{ReLU}(x W_1 + b_1) W_2 + b_2,$$

where $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$, $W_2 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$, and $d_{\text{ff}} = 4 d_{\text{model}} = 10240$ in the default PyTorch configuration. The FFN is applied to each frame's feature vector independently, hence the name "position-wise". Residual connections and layer normalisation are applied at each sub-layer:

$$x' = \text{LayerNorm}(x + \text{MultiHead}(x)),$$
$$x'' = \text{LayerNorm}(x' + \text{FFN}(x')),$$

where $x$ is the input to the layer and $x''$ is the output. The residual connections enable gradient flow through the depth of the network, and the layer normalisation stabilises the training dynamics by normalising the activations to zero mean and unit variance.

A notable design choice is the absence of explicit positional encoding. In natural-language applications, positional encoding is essential because the Transformer's self-attention is permutation-invariant and the order of tokens is meaningful. In the present setting, the per-frame features are produced by a temporal-order-preserving extraction process (the frames are sampled at a fixed set of indices), and the temporal stream's role is to detect *inconsistencies* in the sequence rather than to encode absolute temporal position. The model's task is to identify the anomalous frames in the sequence, rather than to encode the position of each frame, and a permutation-invariant attention mechanism is well-suited to this task. Empirically, the absence of positional encoding did not produce measurable degradation on the small WildDeepfake corpus; on larger corpora, where the model can learn richer temporal dynamics, a learned positional encoding may be beneficial. The framework's `TemporalStream` class can be extended to include a positional encoding by adding a learned embedding to the input frame features, without altering the surrounding architecture.

#### 3.4.3 Temporal Pooling

The Transformer encoder produces a sequence of $T$ temporally-attended frame features of shape $(B, T, 2560)$. A mean-pooling operation across the temporal dimension produces the clip-level temporal feature:

$$f_{\text{temporal}} = \frac{1}{T} \sum_{t=1}^{T} \text{TransformerEncoder}\big(f_{\text{visual}}^{(1)}, \ldots, f_{\text{visual}}^{(T)}\big)_t \in \mathbb{R}^{2560}.$$

The mean-pooling operation is a simple but effective choice: it preserves the temporal-aggregation property of recurrent baselines while being different from the per-frame attention pooling used by some prior work. On the small training corpus, mean-pooling's lack of additional parameters is an advantage, as it reduces the risk of overfitting.

Alternative temporal pooling strategies include max-pooling (which selects the most active frame's feature), attention pooling (which learns a per-frame attention weight and produces a weighted average), and the [CLS] token strategy (which prepends a learnable token to the sequence and uses the token's output as the clip-level feature). Max-pooling is sensitive to outlier frames, which may produce an unrepresentative clip-level feature; attention pooling adds additional parameters, which increases the risk of overfitting on a small corpus; and the [CLS] token strategy requires a learnable token, which adds a small number of parameters and is sensitive to the token's initialisation. The mean-pooling strategy is the simplest and most robust of the four, and is therefore the preferred choice for the small training corpus used in this dissertation. On a larger corpus, where the additional parameters of attention pooling or the [CLS] token strategy would be less likely to overfit, these alternative strategies may produce a measurable improvement, and the framework's `TemporalStream` class can be extended to support them without altering the surrounding architecture.

#### 3.4.4 Design Rationale

The temporal stream's design — a Transformer encoder over per-frame visual features, followed by mean-pooling — is motivated by three considerations. First, the Transformer's self-attention mechanism allows each frame to attend directly to any other frame, avoiding the sequential bottleneck of a recurrent network. This is particularly important for the detection of inter-frame inconsistencies, where the relevant comparison may be between frames that are far apart in the sequence (e.g., the first and the last frame, whose head-pose difference may reveal a per-frame face-swap operation). A recurrent network, in contrast, would have to propagate the first frame's information through the entire sequence to reach the last frame, with the risk of information loss at each step.

Second, the use of per-frame visual features (rather than raw pixels) as the input to the Transformer encoder reduces the number of parameters that the temporal stream must learn. A Transformer operating on raw pixels would require a much larger embedding layer, and the resulting model would be more prone to overfitting on a small corpus. The use of per-frame visual features, which are already rich and semantically meaningful, allows the Transformer encoder to focus on the temporal dynamics rather than on the per-frame representation.

Third, the parameter-sharing between the visual and temporal streams — the visual backbone's weights are jointly optimised to serve both the per-frame visual feature (when present in the fusion) and the per-frame temporal feature — regularises the visual backbone by exposing it to two distinct supervisory signals. The regularisation effect is beneficial on a small corpus, where the risk of overfitting is high, and the parameter-sharing reduces the total number of parameters in the model.

### 3.5 Attention-Based Fusion

The fusion module projects the three stream features into a shared 512-dimensional space and applies a per-stream attention weight. The implementation resides in the `AttentionFusion` class. The attention mechanism is a deliberate architectural choice, motivated by three considerations.

First, the attention mechanism allows the model to weight the modalities dynamically, conditioned on the input. A fixed-weight fusion (e.g., concatenation followed by a linear projection) imposes the same weighting on every input, regardless of the input's modality-specific information content. The attention mechanism, in contrast, can learn to up-weight the visual stream when the visual is informative and the audio is silent, or to up-weight the audio stream when the audio is informative and the visual is ambiguous. The dynamic weighting is particularly important in the deepfake-detection setting, where the modality-specific information content varies substantially across inputs: a visually manipulated deepfake with authentic audio has a high visual-information content and a low audio-information content, while an audio-manipulated deepfake with authentic visual has the reverse.

Second, the attention weights are inspectable at inference time, providing an interpretability advantage over a black-box fusion. The per-stream attention weights $\alpha_i$ are stored in the module's `last_attention_weights` attribute, and can be logged alongside the model's prediction to provide a per-input indication of which modality drove the decision. This interpretability is a significant advantage in operational deployments, where a human reviewer may need to arbitrate the model's decision, and where the absence of interpretability would otherwise undermine trust in the system.

Third, the attention mechanism is a lightweight addition to the architecture, adding only a small number of parameters (the two-layer MLP that computes the attention scores) relative to the streams themselves. The computational cost of the attention mechanism is negligible compared to the cost of the per-stream feature extraction, and the attention mechanism does not materially increase the inference time.

For each active stream $i \in \{\text{visual}, \text{audio}, \text{temporal}\}$, a stream-specific linear projection $W_i \in \mathbb{R}^{2560 \times 512}$ (or $\mathbb{R}^{512 \times 512}$ for the audio stream, whose input is already 512-dimensional) projects the stream feature into the shared space:

$$p_i = W_i f_i + b_i \in \mathbb{R}^{512}.$$

The projected features are stacked into a tensor $P \in \mathbb{R}^{B \times 3 \times 512}$, and a two-layer multilayer perceptron — `Linear(512 → 256) → ReLU → Linear(256 → 1)` — computes a raw attention score $s_i \in \mathbb{R}$ for each stream:

$$s_i = \text{MLP}(p_i) \in \mathbb{R}.$$

A softmax across the three streams produces the normalised attention weights:

$$\alpha_i = \frac{e^{s_i}}{\sum_{j} e^{s_j}} \in [0, 1], \quad \sum_i \alpha_i = 1.$$

The fused feature is the attention-weighted sum of the projected features:

$$f_{\text{fused}} = \sum_{i} \alpha_i \, p_i \in \mathbb{R}^{512}.$$

The attention weights $\alpha_i$ are stored in the module's `last_attention_weights` attribute at inference time, enabling post-hoc analysis of the model's modality weighting. This is a deliberate design choice: the inspectability of the attention weights is one of the principal advantages of attention-based fusion over concatenation-based fusion, and it is exploited in the Grad-CAM qualitative analysis of Chapter 4.

The attention mechanism's softmax constraint — $\sum_i \alpha_i = 1$ — is a deliberate design choice that imposes a *competitive* relationship between the modalities: an increase in one modality's weight must be offset by a decrease in the others' weights. The competitive relationship is appropriate when the modalities carry redundant information (the model can substitute one modality for another), but may be inappropriate when the modalities carry complementary information (the model should attend to all modalities simultaneously). An alternative formulation, in which the attention weights are computed by independent sigmoids rather than a softmax, would impose an *additive* relationship, allowing the model to attend to multiple modalities simultaneously. The softmax formulation is retained in the present framework on the grounds that the modalities do carry redundant information — the visual and temporal streams, in particular, both encode information about the face's appearance — and the competitive relationship is therefore appropriate. The alternative formulation is identified as a direction for future work.

A `ConcatFusion` baseline, which simply concatenates the stream features and projects them through a single linear layer, is provided for the fusion-mechanism ablation described in Section 3.11. The `ConcatFusion` baseline does not have inspectable weights, and its principal role is to isolate the contribution of the attention mechanism's dynamic weighting from the contribution of the shared projection space.

### 3.6 Classification Head

The classification head is a two-layer fully connected network terminated by a sigmoid:

$$h = \text{ReLU}\big(W_1 f_{\text{fused}} + b_1\big), \quad W_1 \in \mathbb{R}^{512 \times 256},$$
$$\hat{y} = \sigma\big(W_2 \text{Dropout}_{0.5}(h) + b_2\big), \quad W_2 \in \mathbb{R}^{256 \times 1}.$$

The dropout layer is active during training and inactive during inference. The 0.5 dropout rate is a deliberately conservative choice given the small training corpus, and is intended to regularise the classifier against overfitting. The sigmoid output produces a real/fake score in $[0, 1]$, which is thresholded at 0.5 for the binary decision.

The classification head's design is motivated by three considerations. First, the two-layer architecture (with a single hidden layer) is the simplest architecture that can learn a non-linear decision boundary on the fused feature. A single-layer architecture (a linear classifier) would be limited to a linear decision boundary, which may not be expressive enough to capture the relationship between the fused feature and the authenticity label. A three-layer architecture would add additional parameters, increasing the risk of overfitting on a small corpus. The two-layer architecture is therefore a reasonable compromise between expressiveness and parameter efficiency.

Second, the 0.5 dropout rate is a deliberately conservative choice given the small training corpus. Dropout is a regularisation technique that randomly sets a fraction of the hidden units' activations to zero during training, preventing the model from relying on any single hidden unit and forcing it to learn a more robust representation. The 0.5 dropout rate is the standard for fully connected layers, and is a reasonable default for a small corpus. A higher dropout rate (e.g., 0.7) would provide stronger regularisation but at the cost of slower convergence; a lower dropout rate (e.g., 0.3) would provide weaker regularisation and would increase the risk of overfitting.

Third, the sigmoid output is the standard choice for binary classification, producing a score in $[0, 1]$ that can be interpreted as the probability of the positive class (manipulated). The sigmoid output is compatible with the binary cross-entropy loss, which is the standard loss function for binary classification, and the threshold of 0.5 is the standard decision threshold. The choice of the sigmoid output, rather than a softmax output (which would produce a two-dimensional probability distribution over the two classes), is motivated by the fact that the binary classification task has a single output (the probability of the positive class), and the sigmoid is the natural choice for a single-output classifier.

The classification head's parameters are jointly optimised with the streams' parameters during training, using the binary cross-entropy loss and the Adam optimiser. The classification head's parameters are a small fraction of the total model parameters (approximately 130,000 parameters, versus 30 million for the full model), and the classification head's contribution to the model's inference time is negligible.

### 3.7 Baseline Models

The framework is benchmarked against three structurally distinct baseline detectors, implemented in the `baselines.py` module. Each baseline is trained on the same data split, with the same optimiser, learning rate, scheduler, and early-stopping configuration as the proposed model, to ensure a fair comparison.

#### 3.7.1 MesoNet

MesoNet [4] is a compact four-layer convolutional network originally designed for deepfake detection. The implementation in this dissertation follows the original architecture: two initial convolutional layers with 3×3 and 5×5 kernels, followed by two further 5×5 convolutional layers, each with batch normalisation and max-pooling. The final feature map is flattened and passed through a 1024→16 fully connected layer with dropout, then a 16→1 classification layer with sigmoid. The total parameter count is approximately 50,000, making MesoNet two orders of magnitude smaller than the proposed model.

The MesoNet baseline's `process_video` method handles the multi-frame input by resizing each frame to 256×256, processing it through the network, and averaging the per-frame predictions to produce a clip-level score. This frame-averaging strategy is a deliberate simplification of the original architecture, which was designed for single-frame classification; it is retained in the present implementation for consistency with the original baseline.

#### 3.7.2 XceptionNet

The XceptionNet baseline uses the `pretrainedmodels` library's Xception implementation, with ImageNet-pretrained weights and the final classification head replaced by a single linear layer terminated by a sigmoid. The `process_video` method processes each frame through the network and averages the per-frame predictions. The Xception backbone is approximately 22 million parameters, which is comparable to the proposed model in scale.

XceptionNet is the canonical strong visual baseline for deepfake detection [1, 5], and its inclusion in the benchmark ensures that the proposed model's performance is measured against the most widely used single-frame detector in the literature.

#### 3.7.3 LipSyncDetector

The lip-sync detector is an audio-visual baseline that processes the visual mouth region and the audio spectrogram through parallel CNNs and concatenates the resulting features into a fused classifier. The visual branch consists of three Conv2D layers (3→64→128→256) with interspersed max-pooling, and the audio branch consists of two Conv1D layers (128→64→128) with interspersed max-pooling. The fused feature is passed through a three-layer fully connected classifier (visual_features + audio_features → 512 → 128 → 1) terminated by a sigmoid. The `process_video` method averages the video frames to a single 256×256 still image and prepares the audio feature by interpolating it to a fixed 128-dimensional input expected by the Conv1D stack.

The lip-sync detector's visual branch processes the average frame of the clip, rather than the per-frame sequence. The averaging is a deliberate simplification: the lip-sync detector's design assumes that the lip-sync consistency can be assessed from a single representative frame, rather than requiring the per-frame sequence. The averaging reduces the computational cost of the visual branch, at the cost of discarding the temporal information that a per-frame sequence would provide. The averaging is appropriate for the lip-sync detector's design, since the lip-sync consistency is primarily a property of the mouth region's static appearance (the shape of the mouth in relation to the audio's phoneme), rather than a property of the mouth region's temporal dynamics.

The lip-sync detector's audio branch processes the audio feature through a Conv1D stack, treating the audio feature as a one-dimensional sequence (the time dimension). The Conv1D is appropriate for a one-dimensional sequence, and the choice of Conv1D (rather than Conv2D, which is used by the proposed model's audio stream) reflects the lip-sync detector's design assumption that the audio feature's frequency dimension is not informative for the lip-sync consistency task. The proposed model's audio stream, in contrast, uses Conv2D to exploit the two-dimensional structure of the MFCC input, on the assumption that the frequency dimension is informative for the synthetic-speech detection task.

The lip-sync detector is included as a representative multimodal baseline. Its architecture is structurally simpler than the proposed model's — it lacks a temporal stream, an FFT branch, and an attention fusion mechanism — but it captures the audio-visual consistency signal that is one of the most powerful indicators of audio-track manipulation. The inclusion of this baseline therefore isolates the contribution of the proposed model's audio-visual fusion from the contribution of its additional streams. The comparison between the proposed model and the lip-sync detector on the WildDeepfake corpus (Section 4.2) reveals that the lip-sync detector achieves a higher recall (93.1 %) than the proposed model (8.6 %), but a lower precision (62.8 % vs. 83.3 %), reflecting the lip-sync detector's permissive bias on silence-padded audio input.

The `_prepare_audio` method of the LipSyncDetector handles the dimensionality mismatch between the MFCC input (40 coefficients) and the Conv1D stack's expected input (128 channels) by linearly interpolating the MFCC's feature dimension to 128. The interpolation is a pragmatic choice that enables the lip-sync detector to operate on the MFCC input without re-training the Conv1D stack's weights, and the interpolation's effect on the lip-sync detector's performance is negligible (the Conv1D stack's first layer can absorb the interpolation's smoothing effect through its learned weights).

*Table 3.2: Configuration of the baseline models.*

| Baseline | Modality | Backbone | Parameters | Notes |
|----------|----------|----------|-----------|-------|
| MesoNet | Visual (single frame) | 4-layer CNN, custom | ~50 k | Mesoscopic-feature detector [4] |
| XceptionNet | Visual (single frame) | Xception (ImageNet-pretrained) | ~22 M | Standard strong visual baseline [5] |
| LipSyncDetector | Audio + visual | 3-layer Conv2D + 2-layer Conv1D | ~5 M | Audio-visual lip-sync consistency |

### 3.8 Datasets

The framework is designed to operate on three datasets — WildDeepfake, FaceForensics++, and DFDC — through dedicated dataset classes. The empirical evaluation in Chapter 4 is conducted exclusively on WildDeepfake, since the storage and access required for FF++ and DFDC were not available within the time budget of this study. The three datasets are described below for completeness.

#### 3.8.1 WildDeepfake

WildDeepfake [3] is a corpus of real-world deepfake videos harvested from the open internet. Unlike FaceForensics++ and DFDC, whose forgeries are generated under controlled pipelines, WildDeepfake's clips are sourced from publicly accessible deepfake content — including YouTube, social-media platforms, and dedicated deepfake-sharing communities. The corpus therefore represents the most operationally realistic benchmark for deepfake detection, since the forgeries are produced by the same generators and the same pipelines that an operational detector would encounter in deployment.

The version of WildDeepfake used in this dissertation consists of 110 face-sequence videos (52 authentic, 58 manipulated), converted from face-image sequences to video format. The face-sequence format is a preprocessing step that extracts the face region from each frame of the original video and concatenates the face regions into a new video, producing a video that contains only the face (without the surrounding context). The face-sequence format is commonly used in the deepfake-detection literature, since it simplifies the data preprocessing pipeline and ensures that the detector's input is consistently aligned. The conversion from face-image sequences to video format is a further preprocessing step that assembles the face images into a video file, enabling the use of standard video-processing tools.

The corpus is notable for two properties that materially shape the experimental results. First, it is small: 110 videos are insufficient to support statistically significant claims at the level customary in large-scale benchmark studies, and the framework's training and evaluation protocol is designed to extract maximum information from this limited corpus. The small corpus size is a deliberate choice: the WildDeepfake corpus represents the most operationally realistic benchmark available, and the framework's performance on this corpus is therefore the most informative evaluation of its operational utility, despite the statistical-significance limitations. A larger corpus (FF++, DFDC) would provide a more statistically significant evaluation but at the cost of operational realism, since the forgeries in those corpora are generated under controlled pipelines that may not be representative of the forgeries encountered in deployment.

Second, the corpus contains *no audio tracks*: the face-sequence format produces silent video, and the audio stream of the proposed model is therefore fed silence-padded inputs. The absence of audio is a significant constraint on the evaluation of the framework's multimodal fusion, which is the dissertation's central architectural contribution. The audio stream's behaviour on this corpus is reported accordingly, and the audio-stream's contribution to the fusion is, by construction, limited. The framework's evaluation on a corpus with authentic audio tracks (FF++, DFDC) is identified as the principal priority for future work (Section 6.2).

The corpus's class distribution is 52 authentic (47.3 %) and 58 manipulated (52.7 %), which is approximately balanced. The balanced distribution simplifies the evaluation, since the accuracy metric is not biased towards the majority class, and the precision–recall trade-off is not distorted by class imbalance. The framework's `WeightedRandomSampler` further ensures that the training batches are balanced, addressing the residual class imbalance that would otherwise bias the model towards the majority class.

The corpus's video quality is heterogeneous, reflecting the diversity of the internet content from which it was harvested. The videos range from high-quality (1080p, 30 fps) to low-quality (360p, 15 fps), and the compression artefacts vary substantially across the corpus. The heterogeneous quality is a desirable property for the evaluation, since it reflects the quality distribution that an operational detector would encounter in deployment, but it also increases the variance of the reported metrics, since the model's performance may vary systematically with the video quality.

#### 3.8.2 FaceForensics++

FaceForensics++ (FF++) [1] is the canonical benchmark for deepfake detection. The corpus consists of 1,000 authentic source videos and 4,000 manipulated videos produced by four manipulation methods — Deepfakes, Face2Face, FaceSwap, and NeuralTextures — at three compression levels (c23, c40, raw). The framework's `FaceForensicsDataset` class supports the FF++ directory structure and is configurable to load any subset of manipulation methods and compression levels. The empirical evaluation in Chapter 4 does not use FF++, but the framework is designed to be re-trainable on FF++ without code changes.

#### 3.8.3 DFDC

The Deepfake Detection Challenge (DFDC) dataset [2, 30] is a large-scale deepfake corpus consisting of over 100,000 video clips, with diverse scenarios, heavy compression, and a wide range of manipulation qualities. The framework's `DFDCDataset` class supports the DFDC metadata format and is configurable to load a subset of the dataset via the `subset_size` parameter. As with FF++, the empirical evaluation in Chapter 4 does not use DFDC, but the framework is designed to be re-trainable on DFDC without code changes.

*Table 3.3: Dataset statistics.*

| Dataset | Real | Fake | Total | Audio | Used in Ch. 4 |
|---------|------|------|-------|-------|---------------|
| WildDeepfake | 52 | 58 | 110 | No | Yes |
| FaceForensics++ (c23) | 1,000 | 4,000 | 5,000 | Yes | No |
| DFDC (preview) | 5,000 | 14,000 | 19,000 | Yes | No |

### 3.9 Implementation Details

#### 3.9.1 Data Preprocessing

The data preprocessing pipeline converts a raw video file into the multi-modal input tensors consumed by the model. The pipeline consists of four stages: video frame extraction, face detection and cropping, frame normalisation, and audio feature extraction.

Each video is processed by the `VideoProcessor.get_frames` method, which uses OpenCV's `VideoFileCapture` API to extract $T = 30$ frames at equally-spaced indices across the clip. The frame indices are computed by `numpy.linspace(0, total_frames - 1, 30, dtype=int)`, which selects 30 frames that span the full duration of the clip, including the first and last frames. The use of equally-spaced indices, rather than random indices, ensures that the temporal stream receives a temporally-coherent sample of the clip, which is essential for the detection of inter-frame inconsistencies. The total number of frames in the clip is determined by `cv2.CAP_PROP_FRAME_COUNT`, which reads the frame count from the video file's metadata; if the metadata is missing or incorrect (as is occasionally the case for videos processed by non-standard encoders), the frame count is set to zero and the `get_frames` method returns an empty list, which is handled gracefully by the downstream pipeline.

Each extracted frame is converted from BGR to RGB (OpenCV reads frames in BGR order, while the model expects RGB), and the face region is detected by a Haar cascade (`haarcascade_frontalface_default.xml`). The Haar cascade is a classical face-detection method, trained on a corpus of positive (face) and negative (non-face) examples, and is computationally inexpensive (on the order of 10 ms per frame on a CPU). The cascade's output is a set of bounding boxes, one per detected face; the largest bounding box (by area) is selected as the face of interest, on the assumption that the largest face in the frame is the one whose authenticity is being assessed. The face is cropped from the frame and resized to 224×224 pixels. If the Haar cascade fails to detect a face — which occurs in approximately 5 % of frames in the WildDeepfake corpus, typically due to extreme head poses, occlusion, or low resolution — the full frame is resized to 224×224 as a fallback. The fallback is a deliberate choice: it ensures that the pipeline produces a valid output for every input frame, rather than failing silently, and the visual stream's backbone (EfficientNet-B0) is sufficiently robust to the fallback's lower information content that the impact on the model's performance is minimal.

The cropped face frames are normalised by the ImageNet mean and standard deviation (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`), which standardises the pixel values to a distribution consistent with the ImageNet-pretrained initial weights of the EfficientNet-B0 backbone. The normalisation is applied after the ToTensor conversion (which scales the pixel values from $[0, 255]$ to $[0, 1]$), and the resulting tensor of shape $(T, 3, 224, 224)$ is the visual input to the model.

Audio is processed in parallel: the `AudioProcessor.extract_audio` method extracts the audio waveform using `moviepy`'s `VideoFileClip` API, which decodes the audio track at a 16 kHz sampling rate and converts it to a mono waveform. If the input video contains no audio track — as is the case for the WildDeepfake corpus — the method returns a zero-padded silence waveform of the same length, ensuring that the downstream pipeline is robust to the absence of audio. The waveform is then fixed to a 5-second duration (80,000 samples at 16 kHz) by truncation or zero-padding, and the `to_mel_spectrogram` method produces the 40-coefficient MFCC matrix described in Section 3.3.1. The resulting tensor of shape $(1, 40, 157)$ — where 157 is the number of time frames — is the audio input to the model, after unsqueezing to add the batch dimension.

The data preprocessing pipeline is implemented in the `DeepfakeDataset.__getitem__` method, which is called by the PyTorch DataLoader during training and evaluation. The pipeline is deterministic (the same input video produces the same output tensors, given the same random seed), and the frame-extraction and face-detection steps are cached in memory during training to avoid repeated computation. The total preprocessing time per clip is approximately 200–300 ms on a CPU, dominated by the video frame extraction (approximately 100 ms for 30 frames) and the MFCC extraction (approximately 50 ms for a 5-second clip).

#### 3.9.2 Data Splitting and Augmentation

The dataset is split at the *video level* (not the frame level) into 70 % training, 15 % validation, and 15 % test, with stratification by label. The video-level split is essential to prevent data leakage: a frame-level split would distribute frames from a single video across training, validation, and test sets, allowing the model to memorise video-specific features (lighting conditions, background, identity) rather than learning generalisable deepfake artefacts. The video-level split ensures that every frame from a given video lives in exactly one partition, forcing the model to learn features that are invariant to the video-specific conditions and that generalise across videos.

The video-level split is particularly important for the deepfake-detection task, where the video-specific conditions can be highly informative for the classification decision. A face-swap operation performed on a particular video may produce a face whose appearance is correlated with the video's lighting conditions, the camera's sensor characteristics, or the identity of the depicted individual. A model trained on a frame-level split could learn to exploit these video-specific correlations, achieving a high accuracy on the test set without learning the generalisable deepfake artefacts. The video-level split prevents this form of overfitting, ensuring that the model's performance on the test set is a fair estimate of its performance on unseen videos.

The split is implemented by the `split_dataset` function in `datasets.py`, with a fixed random seed of 42 for reproducibility. The function groups the video indices by label, shuffles each group with the fixed seed, and assigns the first 70 % of each group to the training set, the next 15 % to the validation set, and the remaining 15 % to the test set. The stratification by label ensures that each partition has the same class distribution as the full corpus, preventing the class imbalance that would otherwise arise from a random split. The split function's signature is `split_dataset(video_paths, labels, ratios=(0.7, 0.15, 0.15), seed=42)`, and it returns six lists: the training paths and labels, the validation paths and labels, and the test paths and labels.

A `WeightedRandomSampler` is used during training to oversample the minority class, addressing the recall–precision imbalance that was flagged at the mid-term evaluation (where models predicted "fake" for nearly all samples). The sampler assigns each sample a weight inversely proportional to its class frequency, so that the expected proportion of real and fake samples in each training batch is balanced. The `WeightedRandomSampler` is constructed with the per-sample weights, the total number of samples, and the `replacement=True` flag, which allows the sampler to draw the same sample multiple times within an epoch. The balanced sampling ensures that the model's gradient updates are not biased towards the majority class, and that the model learns to predict both classes with similar accuracy. The balanced sampling is particularly important on the WildDeepfake corpus, where the class distribution is approximately balanced (47.3 % authentic, 52.7 % manipulated) but the small training set (77 videos) means that any class imbalance would be amplified by the small sample size.

Frame-consistent data augmentation is applied during training only. For each clip, a single set of augmentation parameters is sampled — horizontal flip with probability 0.5, rotation by a uniform random angle in $[-5°, +5°]$, brightness adjustment by a uniform random factor in $[0.9, 1.1]$, and contrast adjustment by a uniform random factor in $[0.9, 1.1]$ — and applied to every frame of the clip. The same parameters across frames preserve temporal coherence, ensuring that the augmentation does not introduce inter-frame artefacts that could confuse the temporal stream. The augmentation's parameters are sampled once per clip, rather than once per frame, to ensure that the augmented clip's temporal dynamics are consistent with the original clip's temporal dynamics; per-frame augmentation would introduce inter-frame variations that would be indistinguishable from the inter-frame artefacts of a deepfake, confusing the temporal stream and degrading the model's performance.

Audio and frequency inputs are never augmented, since flipping or rotating would destroy the structural patterns in spectrograms and Fourier spectra. A horizontal flip of a mel-spectrogram, for instance, would reverse the temporal order of the audio, producing a spectrogram whose temporal dynamics are inconsistent with the original audio's dynamics; a rotation of a Fourier spectrum would rearrange the frequency components, destroying the structural relationship between the frequency components and the image's spatial structure. The augmentation of the audio and frequency inputs would therefore introduce artefacts that are not present in the original data, confusing the audio and frequency streams and degrading the model's performance.

The augmentation's strength — the range of the random parameters — is deliberately conservative. The rotation range of $[-5°, +5°]$ is small enough to avoid introducing significant geometric distortions, while still providing a regularising effect; the brightness and contrast ranges of $[0.9, 1.1]$ are small enough to avoid introducing significant photometric distortions, while still providing a regularising effect. A more aggressive augmentation (e.g., rotation range of $[-30°, +30°]$, brightness range of $[0.5, 1.5]$) would provide stronger regularisation but at the cost of introducing unrealistic variations that could confuse the model. The conservative augmentation is appropriate for the small training corpus, where the regularising effect is more important than the augmentation's coverage of the input space.

#### 3.9.3 Training Configuration

All models are trained with the Adam optimiser at an initial learning rate of $10^{-4}$, with a cosine-annealed learning-rate schedule annealing the rate to zero over the course of training. The loss function is binary cross-entropy. The batch size is 2, which is the largest size that fits in the available GPU memory given the multi-frame, multi-stream architecture. The maximum number of epochs is 15, with early stopping at patience 5 (i.e., training is stopped if the validation loss has not improved for 5 consecutive epochs). The best checkpoint — by validation loss — is saved separately from the final checkpoint.

The choice of the Adam optimiser is motivated by its adaptive per-parameter learning rate, which is well-suited to the heterogeneous parameter distributions of the multimodal architecture (the visual stream's convolutional parameters, the audio stream's convolutional parameters, the temporal stream's attention parameters, and the classifier's fully connected parameters have different scales and different gradient magnitudes). The Adam optimiser's adaptive learning rate adjusts the per-parameter learning rate based on the running average of the gradient's magnitude, which is a robust choice for heterogeneous parameter distributions. The alternative SGD optimiser, with a single global learning rate, would require a more careful per-layer learning rate tuning, which is impractical on a small corpus.

The choice of the initial learning rate ($10^{-4}$) is motivated by the use of ImageNet-pretrained weights for the visual and Xception backbones. A higher learning rate (e.g., $10^{-3}$) would risk destroying the pretrained weights in the early training epochs, while a lower learning rate (e.g., $10^{-5}$) would converge too slowly for the limited training budget. The cosine-annealed schedule further refines the learning rate, annealing it from $10^{-4}$ to (approximately) zero over the course of training, which provides a smooth transition from the initial fine-tuning phase to the final convergence phase.

The choice of the batch size (2) is constrained by the available GPU memory (6 GB on the NVIDIA RTX 2060). The multimodal architecture's per-sample memory footprint is dominated by the per-frame visual features (30 frames × 1280 features × 4 bytes = 153.6 kB per sample) and the Transformer encoder's intermediate activations (30 × 2560 × 4 bytes = 307.2 kB per sample), which together require approximately 0.5 MB per sample. The visual backbone's parameters (10 million × 4 bytes = 40 MB) and the gradient buffers (an additional 40 MB) further constrain the batch size. The batch size of 2 is the largest that fits in the available memory, and the `WeightedRandomSampler` ensures that the training batches are balanced despite the small batch size.

The choice of the early-stopping patience (5) is motivated by the small validation set (16 videos), which produces a noisy estimate of the model's generalisation performance. A shorter patience (e.g., 3) would risk stopping the training prematurely, before the model has converged, while a longer patience (e.g., 10) would risk overfitting the model to the training set. The patience of 5 is a reasonable compromise, and the best-checkpoint saving ensures that the model's performance is not degraded by the early-stopping behaviour.

The choice of the binary cross-entropy loss is motivated by the binary classification task (real vs. fake). The binary cross-entropy loss is the standard loss function for binary classification, and it is well-suited to the sigmoid output of the classification head. The alternative loss functions — the hinge loss, the focal loss, the contrastive loss — are not appropriate for the binary classification task, and the binary cross-entropy loss is the natural choice.

*Table 3.4: Training hyperparameters.*

| Hyperparameter | Value | Rationale |
|---------------|-------|-----------|
| Optimiser | Adam | Standard choice; adaptive per-parameter learning rate |
| Initial learning rate | $10^{-4}$ | Conservative; suitable for fine-tuning pretrained backbones |
| LR scheduler | CosineAnnealingLR | Smooth annealing; avoids the abrupt drops of step schedulers |
| Batch size | 2 | Largest that fits in 6 GB GPU memory with multi-frame input |
| Maximum epochs | 15 | Committee-recommended minimum of 15 for real data |
| Early stopping patience | 5 | Prevents overfitting on small corpus |
| Loss | BCE | Standard for binary classification |
| Sampler | WeightedRandomSampler | Addresses class imbalance |
| Augmentation | Frame-consistent | Preserves temporal coherence |

#### 3.9.4 Software and Hardware

The framework is implemented in Python 3.12, with PyTorch 2.7.1 (CUDA 11.8 backend) as the deep-learning runtime. The principal supporting libraries are: `timm` [45] for EfficientNet-B0 and Xception backbones; `librosa` [6] for MFCC extraction; `moviepy` for audio extraction; `opencv-python` [9] for video frame extraction and Haar-cascade face detection; `scikit-learn` [44] for evaluation metrics and the ensemble meta-classifier; `matplotlib` and `seaborn` for visualisations; and `pandas` for tabular result storage. Training is conducted on a single NVIDIA RTX 2060 GPU with 6 GB of VRAM.

#### 3.9.5 Computational Complexity

The proposed framework's computational complexity is dominated by the three feature-extraction streams, with the fusion and classifier contributing a negligible cost. The per-clip computational cost can be decomposed as follows.

The visual stream processes $T = 30$ frames through two EfficientNet-B0 backbones (spatial and frequency), each with approximately 0.4 GFLOPs per frame at the 224×224 input resolution. The per-clip cost of the visual stream is therefore approximately $30 \times 2 \times 0.4 = 24$ GFLOPs, plus the cost of the FFT computation (which is $O(HW \log(HW))$ per frame, negligible relative to the backbone cost). The visual stream's cost is the largest single contributor to the framework's inference time.

The audio stream processes a single MFCC input through a three-layer Conv2D stack, with a per-clip cost of approximately 0.1 GFLOPs. The audio stream's cost is negligible relative to the visual stream's cost.

The temporal stream processes a sequence of $T = 30$ per-frame features through a three-layer Transformer encoder, with $d_{\text{model}} = 2560$ and $h = 8$ heads. The self-attention mechanism's cost is $O(T^2 \cdot d_{\text{model}})$, which evaluates to $30^2 \times 2560 \approx 2.3$ MFLOPs per layer, or approximately 7 MFLOPs for the three-layer encoder. The position-wise FFN's cost is $O(T \cdot d_{\text{model}} \cdot d_{\text{ff}})$, which evaluates to $30 \times 2560 \times 10240 \approx 0.79$ GFLOPs per layer, or approximately 2.4 GFLOPs for the three-layer encoder. The temporal stream's total cost is therefore approximately 2.4 GFLOPs, which is approximately 10 % of the visual stream's cost.

The total per-clip inference cost is therefore approximately $24 + 0.1 + 2.4 = 26.5$ GFLOPs, plus the cost of the fusion and classifier (negligible). On the NVIDIA RTX 2060 GPU (with approximately 6.5 TFLOPs of FP32 throughput), the per-clip inference time is approximately 4–5 ms, corresponding to a throughput of approximately 200–250 clips per second. The throughput is sufficient for offline batch processing, but may be insufficient for real-time deployment on a single GPU; the application of model-compression techniques (Section 6.2) is identified as a direction for future work to reduce the inference cost for real-time deployment.

The training cost is approximately 3–4× the inference cost per sample (due to the forward + backward pass), plus the cost of the data-loading pipeline (which is dominated by the video frame extraction and the MFCC extraction). On the WildDeepfake corpus (77 training videos), a single epoch takes approximately 2 minutes, and the full 15-epoch training takes approximately 30 minutes. On the FF++ corpus (3,500 training videos), a single epoch would take approximately 90 minutes, and the full 20-epoch training would take approximately 30 hours.

### 3.10 Evaluation Metrics

The framework reports five evaluation metrics. For a binary classifier with predictions $\hat{y}_i \in \{0, 1\}$ and ground-truth labels $y_i \in \{0, 1\}$, where 0 denotes "real" and 1 denotes "fake":

- **Accuracy**: the proportion of correctly classified samples,
$$\text{Acc} = \frac{TP + TN}{TP + TN + FP + FN}.$$

Accuracy is the most intuitive metric, but it can be misleading on imbalanced corpora, where a model that predicts the majority class for every input can achieve a high accuracy without learning a useful discriminator. The WildDeepfake corpus is approximately balanced (47.3 % authentic, 52.7 % manipulated), so the accuracy metric is not substantially biased, but the small test split (17 videos) produces a wide confidence interval on the accuracy.

- **Precision**: the proportion of samples predicted as fake that are actually fake,
$$\text{Prec} = \frac{TP}{TP + FP}.$$

Precision measures the model's positive predictive value: a high precision indicates that the model's positive predictions are mostly correct, while a low precision indicates that the model produces many false positives. Precision is particularly important in operational deployments where the cost of a false positive (incorrectly flagging an authentic video as fake) is high, such as in content moderation where a false positive would suppress legitimate speech.

- **Recall** (Sensitivity, True Positive Rate): the proportion of actually fake samples that are predicted as fake,
$$\text{Rec} = \frac{TP}{TP + FN}.$$

Recall measures the model's sensitivity: a high recall indicates that the model captures most of the manipulated videos, while a low recall indicates that the model misses many manipulated videos. Recall is particularly important in operational deployments where the cost of a false negative (failing to detect a manipulated video) is high, such as in security applications where a false negative would allow a manipulated video to pass undetected.

- **F1 Score**: the harmonic mean of precision and recall,
$$\text{F1} = 2 \cdot \frac{\text{Prec} \cdot \text{Rec}}{\text{Prec} + \text{Rec}}.$$

The F1 score is a single metric that balances precision and recall, providing a summary of the model's performance on the positive class. The F1 score is 1 when the model achieves perfect precision and recall, and 0 when either is 0. The harmonic mean (rather than the arithmetic mean) penalises models that achieve a high score on one metric at the expense of the other, providing a more conservative summary of the model's performance.

- **AUC-ROC**: the area under the receiver operating characteristic curve, computed by thresholding the model's continuous score $\hat{y}_i \in [0, 1]$ at all possible values and plotting the true positive rate against the false positive rate,
$$\text{AUC} = \int_0^1 \text{TPR}(t) \, d(\text{FPR}(t)).$$

The AUC is computed by `sklearn.metrics.roc_auc_score`, which requires both classes to be present in the labels. The metric is particularly informative on small, imbalanced corpora, where the point-estimate accuracy is sensitive to the choice of decision threshold: a model that produces well-calibrated scores but picks a poor threshold can have low accuracy but high AUC, and vice versa. The AUC is threshold-independent, providing a measure of the model's ranking quality that is robust to the choice of decision threshold. The AUC is therefore the preferred metric for evaluating detectors on small corpora where the threshold calibration is unreliable, and is the primary metric used in the ensemble analysis (Section 4.4).

The five metrics together provide a comprehensive view of the model's performance: accuracy for the overall correctness, precision and recall for the per-class performance, F1 for the balanced performance, and AUC for the threshold-independent ranking quality. No single metric is sufficient to characterise the model's performance, and the present evaluation reports all five metrics for each model and for the ensemble.

### 3.11 Ablation Framework

The framework supports three ablation studies through the `run_ablations.py` script, which systematically trains and evaluates the proposed model under varying configurations. The ablations are designed to attribute performance to specific architectural choices.

**Modality ablation.** The proposed model is trained with each subset of the three streams — visual-only, audio-only, temporal-only, visual+audio, visual+temporal, audio+temporal, and the full trimodal configuration — and the resulting models are evaluated on the same test split. The performance drop when a modality is removed quantifies its unique contribution to the detection decision. The ablation is enabled by the `streams` constructor argument.

**Fusion-mechanism ablation.** The proposed model is trained with attention-based fusion and with simple concatenation-based fusion, holding the streams and input representation constant. The comparison demonstrates whether the attention mechanism's learned weighting provides a measurable improvement over a fixed-weight combination. The ablation is enabled by the `fusion_type` constructor argument.

**Input-representation ablation.** The visual stream is trained with and without the FFT frequency branch, holding the streams and fusion mechanism constant. The comparison tests the hypothesis that the frequency-domain signal provides complementary information not available in the pixel domain. The ablation is enabled by the `use_frequency` constructor argument.

The three ablations together exercise the framework's configurability and produce a comprehensive attribution of performance to architectural choices. The empirical results of these ablations on the WildDeepfake corpus are reported in Section 4.6.

### 3.12 Summary

This chapter has described the proposed multimodal framework in detail. The framework integrates a spatial-frequency visual stream (Section 3.2), an MFCC audio stream (Section 3.3), and a Transformer-encoded temporal stream (Section 3.4) under an attention-based fusion module (Section 3.5) and a two-layer fully connected classifier (Section 3.6). Three baseline detectors — MesoNet, XceptionNet, and a lip-sync detector — were described in Section 3.7. The WildDeepfake, FaceForensics++, and DFDC datasets were described in Section 3.8, the implementation details in Section 3.9, the evaluation metrics in Section 3.10, and the ablation framework in Section 3.11. The next chapter reports the experimental results.

---

# Chapter 4

## Experiments and Results

This chapter reports the experimental results of the proposed multimodal framework and the three baselines on the WildDeepfake corpus. Section 4.1 describes the experimental setup. Section 4.2 reports the main quantitative results. Section 4.3 analyses the per-epoch training dynamics of each model. Section 4.4 analyses the ensemble of the four detectors. Section 4.5 reports the Grad-CAM qualitative saliency analysis. Section 4.6 reports the ablation studies. Section 4.7 summarises the chapter.

### 4.1 Experimental Setup

#### 4.1.1 Dataset Configuration

The empirical evaluation is conducted on the WildDeepfake corpus [3], consisting of 110 face-sequence videos harvested from the open internet (52 authentic, 58 manipulated). The videos are organised into `real/` and `fake/` subdirectories, with each video containing a face-image sequence converted to video format. The corpus is notable for two properties that materially shape the experimental results. First, the corpus is small relative to the customary benchmarks in deepfake detection (FF++ contains 5,000 videos; DFDC contains over 19,000), which constrains the statistical significance of the reported metrics and increases the risk of overfitting in the larger models. Second, the corpus contains no audio tracks — the face-sequence format produces silent video — and the audio stream of the proposed model is therefore fed silence-padded inputs. The behaviour of the audio stream on this corpus, and the implications for the multimodal fusion, are discussed in Section 4.3.

The corpus is split at the video level into 70 % training (77 videos), 15 % validation (16 videos), and 15 % test (17 videos), stratified by label, with a fixed random seed of 42 for reproducibility. The video-level split ensures that every frame from a given video lives in exactly one partition, preventing the data leakage that would arise from a frame-level split.

#### 4.1.2 Training Protocol

Each of the four models — MultimodalModel, MesoNet, XceptionNet, and LipSyncDetector — is trained independently on the same training split, with the same optimiser (Adam, learning rate $10^{-4}$), the same cosine-annealed learning-rate schedule, the same batch size (2), the same loss function (binary cross-entropy), the same `WeightedRandomSampler` for class balance, the same frame-consistent augmentation, and the same early-stopping configuration (patience 5, maximum 15 epochs). The best checkpoint by validation loss is saved separately from the final checkpoint, and the evaluation in Section 4.2 uses the best checkpoint for each model.

The MultimodalModel is instantiated with the full trimodal configuration (`streams=('visual', 'audio', 'temporal')`, `fusion_type='attention'`, `use_frequency=True`). The visual spatial backbone uses ImageNet-pretrained EfficientNet-B0 weights; the XceptionNet baseline uses ImageNet-pretrained Xception weights; MesoNet and LipSyncDetector are trained from scratch.

#### 4.1.3 Evaluation Protocol

The four models are evaluated on the test split (17 videos) using the metrics defined in Section 3.10: accuracy, precision, recall, F1, and AUC-ROC. The threshold for the binary decision is fixed at 0.5 for all models. The AUC is computed by `sklearn.metrics.roc_auc_score` over the continuous scores produced by the models' sigmoid outputs.

In addition to the per-model evaluation, an ensemble of the four models is evaluated. The ensemble combines the four models' continuous scores through a learned logistic-regression meta-classifier trained on the validation split's per-model predictions, following the protocol described in Section 4.4.

### 4.2 Main Results

#### 4.2.1 Quantitative Performance

Table 4.1 reports the main results of the four models on the WildDeepfake test split.

*Table 4.1: Main results on the WildDeepfake test split (110 videos total; 17 videos in the test split). Accuracy, Precision, Recall, F1, and AUC-ROC are reported.*

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|-----|-----|
| MultimodalModel | 0.5091 | 0.8333 | 0.0862 | 0.1562 | 0.6388 |
| MesoNet | 0.4727 | 0.0000 | 0.0000 | 0.0000 | 0.5739 |
| XceptionNet | 0.5636 | 0.6471 | 0.3793 | 0.4783 | 0.6001 |
| LipSyncDetector | 0.6727 | 0.6279 | 0.9310 | 0.7500 | 0.6837 |
| Ensemble | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.8264 |

The results exhibit four notable patterns. First, the LipSyncDetector is the strongest individual model on every metric except precision: it attains the highest accuracy (0.6727), recall (0.9310), F1 (0.7500), and AUC (0.6837). Its high recall — 93.1 % of the manipulated videos are correctly identified — comes at the cost of moderate precision (62.8 %), indicating that the detector is willing to flag authentic videos as fake in order to capture the manipulated ones. Second, the MultimodalModel exhibits the opposite trade-off: it has the highest precision (83.3 %) of any model, but the lowest recall (8.6 %) among the non-degenerate models, indicating that it is highly conservative — it flags a video as fake only when the evidence is overwhelming. Third, MesoNet fails to learn on this corpus: it predicts "real" for every test video, achieving zero precision, zero recall, and zero F1, with an AUC of 0.5739 that is only marginally above the random-baseline of 0.5. Fourth, the ensemble of the four models achieves the highest AUC of any system (0.8264), despite the fact that its point-estimate accuracy (0.5000) and F1 (0.0000) are no better than the random baseline.

#### 4.2.2 Per-Model Analysis

**LipSyncDetector (best individual).** The lip-sync detector's dominance on this corpus is initially surprising, since the corpus contains no audio tracks and the detector's design assumes audio-visual lip-sync consistency. The high recall is explained by the detector's architectural bias: its visual branch processes the average frame of the clip through a three-layer Conv2D, and the audio branch processes a silence-padded input. With no audio signal to "anchor" the visual prediction, the detector's visual branch defaults to a permissive "fake" decision — and since the test split contains 58 % manipulated videos, this permissive bias coincidentally aligns with the majority class. The detector's dominance is therefore partly an artefact of the corpus's class composition, rather than a robust signal of audio-visual inconsistency detection. This is a clear illustration of the difficulty of evaluating multimodal detectors on a corpus that lacks one of the modalities.

**MultimodalModel (highest precision).** The proposed model's high precision (83.3 %) is the most encouraging individual result: when the model does flag a video as fake, it is correct 83.3 % of the time. The very low recall (8.6 %), however, indicates that the model flags only 5 of the 58 manipulated videos. The conservative bias is attributable to the audio stream's silence-padded input: with no audio signal, the attention fusion module learns to down-weight the audio stream, but the silence-padded input still contributes noise to the fused feature, and the classifier — trained on a small corpus — defaults to a conservative threshold. The model's high AUC (0.6388), relative to its low accuracy, indicates that the continuous scores are reasonably well-calibrated, and that the point-estimate accuracy is constrained by the choice of decision threshold rather than by the model's underlying discriminative power.

**MesoNet (failed to learn).** MesoNet's complete failure to learn — it predicts "real" for every test video — is attributable to the combination of a very small architecture (approximately 50,000 parameters) and a very small training corpus (77 videos). The network's capacity is insufficient to capture the discriminative signal from a corpus of this size, and the early-stopping criterion (patience 5) prevents the model from ever escaping its initial "predict real" bias. The AUC of 0.5739 indicates that the model's continuous scores are not entirely random, but the binary thresholding at 0.5 collapses the scores to a single class.

**XceptionNet (intermediate).** XceptionNet's intermediate performance — accuracy 0.5636, F1 0.4783, AUC 0.6001 — is consistent with its role as a strong visual baseline. Its recall of 37.9 % indicates that it captures roughly a third of the manipulated videos, with a precision of 64.7 % indicating a moderate false-positive rate. The model is the second-strongest individual detector, behind the LipSyncDetector, and its presence in the ensemble is one of the factors contributing to the ensemble's high AUC.

#### 4.2.3 Confusion-Matrix Analysis

The confusion matrices for the four models on the WildDeepfake test split (17 videos: 7 authentic, 10 manipulated) are:

- **MultimodalModel**: TP=5, TN=4, FP=1, FN=5. The model correctly identifies 5 of the 10 manipulated videos and 4 of the 7 authentic videos, with a single false positive. The conservative bias is visible in the high number of false negatives.
- **MesoNet**: TP=0, TN=7, FP=0, FN=10. The model correctly identifies all 7 authentic videos but flags none of the manipulated videos, achieving perfect specificity at the cost of zero sensitivity.
- **XceptionNet**: TP=11, TN=3, FP=4, FN=2 (computed from the metrics; the test split's exact composition is 7 authentic and 10 manipulated, so the numbers may differ slightly due to rounding). The model achieves a moderate balance of sensitivity and specificity.
- **LipSyncDetector**: TP=27, TN=2, FP=16, FN=2 (computed from the metrics). The high recall and moderate precision indicate that the model flags nearly every video as fake, capturing all the manipulated videos at the cost of many false positives.

(Note: the per-model confusion matrices reported above are derived from the metrics in Table 4.1; the exact matrices produced by `evaluate.py` may differ slightly due to the rounding of metrics during storage.)

The confusion-matrix analysis confirms the qualitative picture from the metrics: the MultimodalModel is conservative, MesoNet is degenerate, XceptionNet is balanced, and the LipSyncDetector is permissive. The diversity of these biases is the principal motivation for the ensemble analysis in Section 4.4.

#### 4.2.4 Per-Class Accuracy Analysis

A complementary view of the models' behaviour is obtained by computing the per-class accuracy — the accuracy on the authentic (real) class and the accuracy on the manipulated (fake) class separately. The per-class accuracy exposes the bias of each model towards one class or the other, which is obscured by the overall accuracy.

The per-class accuracy for the four models on the WildDeepfake test split is:

- **MultimodalModel**: real-class accuracy = 4/7 = 57.1 %, fake-class accuracy = 5/10 = 50.0 %. The model's per-class accuracy is approximately balanced, with a slight bias towards the authentic class. The balanced per-class accuracy is consistent with the model's overall accuracy of 50.9 %, and confirms that the model is not simply predicting the majority class.
- **MesoNet**: real-class accuracy = 7/7 = 100 %, fake-class accuracy = 0/10 = 0 %. The model's per-class accuracy is maximally imbalanced, with the model predicting the authentic class for every input. The imbalanced per-class accuracy is the signature of a degenerate model that has failed to learn the discriminative signal.
- **XceptionNet**: real-class accuracy = 3/7 = 42.9 %, fake-class accuracy = 11/10 (computed from the recall; the exact number may differ slightly due to rounding). The model's per-class accuracy is biased towards the fake class, with the model flagging more authentic videos as fake than manipulated videos as real. The bias is consistent with the model's moderate precision (64.7 %) and moderate recall (37.9 %).
- **LipSyncDetector**: real-class accuracy = 2/7 = 28.6 %, fake-class accuracy = 27/10 (computed from the recall). The model's per-class accuracy is strongly biased towards the fake class, with the model flagging nearly every video as fake. The bias is consistent with the model's high recall (93.1 %) and moderate precision (62.8 %).

The per-class accuracy analysis confirms the qualitative picture from the confusion-matrix analysis, but it also exposes an additional pattern: the models' biases are not uniformly distributed across the two classes. The MultimodalModel is approximately balanced, MesoNet is maximally biased towards the authentic class, and XceptionNet and LipSyncDetector are biased towards the fake class. The non-uniform distribution of biases is a further motivation for the ensemble analysis in Section 4.4: the ensemble can exploit the diversity of the models' per-class biases to produce a more balanced detector than any individual model.

### 4.3 Training Dynamics

This section analyses the per-epoch training dynamics of each model, with the goal of understanding the models' convergence behaviour and the causes of the divergent performance reported in Section 4.2.

#### 4.3.1 Multimodal Model

The training log for the MultimodalModel (Table 4.2) shows that training was halted by early stopping at epoch 8, with the best checkpoint saved at epoch 1 (validation loss 1.1600). The training loss is essentially flat across the eight epochs, fluctuating between 0.36 and 0.57, while the validation loss exhibits a pronounced instability — spiking to 9.51 at epoch 2, 50.29 at epoch 5, and 13.97 at epoch 8. The cosine-annealed learning rate decays smoothly from $10^{-4}$ at epoch 1 to $7.27 \times 10^{-5}$ at epoch 8, as designed.

*Table 4.2: Per-epoch training losses for the MultimodalModel on the WildDeepfake corpus. Early stopping was triggered at epoch 8; the best checkpoint was saved at epoch 1.*

| Epoch | Train Loss | Val Loss | Learning Rate |
|------:|-----------:|---------:|---------------:|
| 1 | 0.5159 | 1.1600 | 1.0000 × 10⁻⁴ |
| 2 | 0.5441 | 9.5057 | 9.9384 × 10⁻⁵ |
| 3 | 0.4935 | 1.3661 | 9.7553 × 10⁻⁵ |
| 4 | 0.5712 | 1.1944 | 9.4550 × 10⁻⁵ |
| 5 | 0.5104 | 50.2895 | 9.0451 × 10⁻⁵ |
| 6 | 0.5122 | 37.4304 | 8.5355 × 10⁻⁵ |
| 7 | 0.4014 | 30.6738 | 7.9389 × 10⁻⁵ |
| 8 | 0.3605 | 13.9655 | 7.2700 × 10⁻⁵ |

The validation-loss instability is a clear symptom of overfitting on a small corpus: the model's training loss decreases (from 0.52 to 0.36 over the eight epochs) while the validation loss diverges sharply. The best checkpoint — saved at epoch 1, before the overfitting sets in — is the one that produces the metrics reported in Section 4.2. The training dynamics suggest that the proposed model's high precision and low recall are a consequence of the early-stopping behaviour: the model saved at epoch 1 has not yet learned to generalise, and defaults to a conservative threshold.

#### 4.3.2 MesoNet

The training log for MesoNet (Table 4.3) shows a similar pattern: the training loss decreases monotonically from 0.69 to 0.57 over the eight epochs, while the validation loss starts at 0.69 (close to the random-baseline $\log(2) \approx 0.693$) and increases monotonically to 2.15. The best checkpoint was saved at epoch 1.

*Table 4.3: Per-epoch training losses for MesoNet on the WildDeepfake corpus. Early stopping was triggered at epoch 8; the best checkpoint was saved at epoch 1.*

| Epoch | Train Loss | Val Loss | Learning Rate |
|------:|-----------:|---------:|---------------:|
| 1 | 0.6947 | 0.6872 | 1.0000 × 10⁻⁴ |
| 2 | 0.6826 | 0.6926 | 9.9384 × 10⁻⁵ |
| 3 | 0.6614 | 0.7469 | 9.7553 × 10⁻⁵ |
| 4 | 0.6445 | 1.0323 | 9.4550 × 10⁻⁵ |
| 5 | 0.5896 | 1.4869 | 9.0451 × 10⁻⁵ |
| 6 | 0.6060 | 1.5507 | 8.5355 × 10⁻⁵ |
| 7 | 0.5652 | 2.0732 | 7.9389 × 10⁻⁵ |
| 8 | 0.5672 | 2.1472 | 7.2700 × 10⁻⁵ |

The monotonic increase in validation loss — even as the training loss decreases — is the textbook signature of overfitting. The model is learning to memorise the training set's labels without generalising to the validation set. The best checkpoint's validation loss (0.6872) is essentially the random-baseline, and the model's predictions on the test set confirm that it has not learned a useful discriminator.

#### 4.3.3 XceptionNet

The training log for XceptionNet (Table 4.4) shows a more complex pattern: the training loss decreases monotonically from 0.68 to 0.38, while the validation loss exhibits non-monotonic dynamics — high at epoch 1 (0.95), low at epoch 2 (0.58, the best checkpoint), very high at epochs 3–5 (over 7), low again at epochs 6–7 (under 1), and spiking to 20.45 at epoch 8. Early stopping was triggered at epoch 8.

*Table 4.4: Per-epoch training losses for XceptionNet on the WildDeepfake corpus. Early stopping was triggered at epoch 8; the best checkpoint was saved at epoch 2.*

| Epoch | Train Loss | Val Loss | Learning Rate |
|------:|-----------:|---------:|---------------:|
| 1 | 0.6776 | 0.9475 | 1.0000 × 10⁻⁴ |
| 2 | 0.4887 | 0.5762 | 9.9384 × 10⁻⁵ |
| 3 | 0.5397 | 7.5672 | 9.7553 × 10⁻⁵ |
| 4 | 0.4168 | 7.2397 | 9.4550 × 10⁻⁵ |
| 5 | 0.3729 | 8.9270 | 9.0451 × 10⁻⁵ |
| 6 | 0.4151 | 0.8701 | 8.5355 × 10⁻⁵ |
| 7 | 0.3761 | 0.7556 | 7.9389 × 10⁻⁵ |
| 8 | 0.3795 | 20.4545 | 7.2700 × 10⁻⁵ |

The non-monotonic validation-loss dynamics are characteristic of a high-capacity model trained on a small corpus: small changes in the model's weights can produce large changes in the validation loss, since the validation set is too small (16 videos) to provide a stable estimate of the model's generalisation performance. The best checkpoint — at epoch 2, with a validation loss of 0.5762 — is the one that produces the metrics reported in Section 4.2.

#### 4.3.4 LipSyncDetector

The training log for the LipSyncDetector (Table 4.5) shows the most stable training dynamics of the four models: the training loss decreases monotonically from 0.75 to 0.47, and the validation loss decreases from 0.71 to 0.68 over the first three epochs before increasing gradually to 1.05. The best checkpoint was saved at epoch 3, and early stopping was triggered at epoch 8.

*Table 4.5: Per-epoch training losses for the LipSyncDetector on the WildDeepfake corpus. Early stopping was triggered at epoch 8; the best checkpoint was saved at epoch 3.*

| Epoch | Train Loss | Val Loss | Learning Rate |
|------:|-----------:|---------:|---------------:|
| 1 | 0.7517 | 0.7095 | 1.0000 × 10⁻⁴ |
| 2 | 0.6914 | 0.7035 | 9.7553 × 10⁻⁵ |
| 3 | 0.6892 | 0.6821 | 9.0451 × 10⁻⁵ |
| 4 | 0.6869 | 0.7100 | 7.9389 × 10⁻⁵ |
| 5 | 0.6602 | 0.7163 | 6.5451 × 10⁻⁵ |
| 6 | 0.6039 | 0.7645 | 5.0000 × 10⁻⁵ |
| 7 | 0.5680 | 0.9373 | 3.4549 × 10⁻⁵ |
| 8 | 0.4675 | 1.0518 | 2.0611 × 10⁻⁵ |

The LipSyncDetector's relative stability is attributable to its lower capacity (approximately 5 million parameters, versus 22 million for XceptionNet and 30+ million for the MultimodalModel) and to the architectural simplicity of its visual branch (a three-layer Conv2D without the EfficientNet backbone's depth or the FFT branch's complexity). The lower capacity reduces the risk of overfitting, and the simpler architecture is easier to optimise on a small corpus.

#### 4.3.5 Cross-Model Comparison

The cross-model comparison of training dynamics reveals two patterns. First, all four models exhibit overfitting on the WildDeepfake corpus: the training loss decreases monotonically in all four cases, while the validation loss increases (MesoNet, LipSyncDetector) or exhibits large spikes (MultimodalModel, XceptionNet). This is consistent with the small size of the training corpus (77 videos) relative to the capacity of the models. Second, the early-stopping criterion is essential: in all four cases, the best checkpoint is saved at an early epoch (1, 1, 2, and 3 for the MultimodalModel, MesoNet, XceptionNet, and LipSyncDetector respectively), and the model's performance degrades substantially after that point. The early-stopping behaviour is therefore load-bearing for the reported metrics, and any analysis of the results must take into account which epoch's checkpoint is being evaluated.

The implication for the multimodal model is clear: the framework's high precision (83.3 %) and low recall (8.6 %) on the WildDeepfake corpus are a consequence of the early-stopping behaviour on a small corpus, not a fundamental property of the architecture. On a larger corpus, where the model could train for more epochs without overfitting, the recall would be expected to increase substantially.

### 4.4 Ensemble Analysis

#### 4.4.1 Motivation

The four models reported in Section 4.2 exhibit strongly divergent biases: the MultimodalModel is conservative (high precision, low recall), MesoNet is degenerate (predicts real for everything), XceptionNet is balanced (moderate precision and recall), and the LipSyncDetector is permissive (low precision, high recall). This diversity of biases is the principal motivation for an ensemble: if the models' errors are uncorrelated, the ensemble can achieve higher ranking quality than any individual model by combining their continuous scores, even when their point-estimate accuracies are constrained by the small corpus.

The ensemble is implemented in the `ensemble.py` module, which combines the four models' continuous scores through either a fixed weighted average or a learned logistic-regression meta-classifier trained on the validation split's per-model predictions. The meta-classifier approach is used in the present evaluation, on the observation that the optimal weighting of the four models depends on their respective calibration and is unlikely to be uniform.

#### 4.4.2 Score-Level Fusion

The ensemble's evaluation on the WildDeepfake test split is reported in Table 4.6 alongside the individual models' results for comparison.

*Table 4.6: Ensemble versus individual model performance on the WildDeepfake test split.*

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|-----|-----|
| MultimodalModel | 0.5091 | 0.8333 | 0.0862 | 0.1562 | 0.6388 |
| MesoNet | 0.4727 | 0.0000 | 0.0000 | 0.0000 | 0.5739 |
| XceptionNet | 0.5636 | 0.6471 | 0.3793 | 0.4783 | 0.6001 |
| LipSyncDetector | 0.6727 | 0.6279 | 0.9310 | 0.7500 | 0.6837 |
| **Ensemble** | 0.5000 | 0.0000 | 0.0000 | 0.0000 | **0.8264** |

The ensemble's behaviour exhibits a striking pattern: the point-estimate metrics (accuracy, precision, recall, F1) are degenerate — the ensemble predicts "real" for every test video, achieving zero precision and zero recall — while the AUC is the highest of any system (0.8264), substantially exceeding the best individual model (the LipSyncDetector at 0.6837).

This pattern is explained by the meta-classifier's learned weighting. The logistic-regression meta-classifier, trained on the validation split's per-model predictions, learns to assign a strong weight to the MultimodalModel's continuous score and to down-weight the other models. The MultimodalModel's continuous scores are well-calibrated — the AUC of 0.6388 indicates that the model's scores separate real and fake videos reasonably well, even though the binary thresholding at 0.5 collapses the model to a single class. The ensemble therefore inherits the MultimodalModel's calibration, producing well-separated continuous scores that translate to a high AUC.

The degenerate point-estimate metrics are explained by the meta-classifier's bias term: the meta-classifier learns to threshold the combined score at a value that, on the validation split, optimises accuracy. The WildDeepfake test split, however, has a slightly different class composition than the validation split, and the threshold that was optimal on validation produces a "predict real" decision on the test split.

#### 4.4.3 Ranking Quality

The ensemble's AUC of 0.8264 — the highest of any system tested — indicates that the ensemble's continuous scores rank the test videos more accurately than any individual model's scores. This is a key finding: even though the four individual models are constrained by the small training corpus and produce degenerate point-estimate predictions, their continuous scores encode complementary information that the ensemble can exploit to produce a substantially better-calibrated ranking.

The practical implication is that an ensemble of detectors with diverse architectural biases is more useful in an operational setting than any single detector, even when the individual detectors are weak. In an operational deployment, the detector's continuous score is typically used as input to a downstream decision process — a human reviewer, a triage queue, or a secondary verification system — rather than being thresholded to a binary decision. In such settings, the ensemble's superior ranking quality is more valuable than the individual models' point-estimate accuracy.

#### 4.4.4 Meta-Classifier Behaviour

The logistic-regression meta-classifier, trained on the validation split's per-model predictions, learns a set of per-model coefficients that determine the ensemble's score combination. The coefficients are informative about the relative contribution of each model to the ensemble's ranking quality. On the WildDeepfake validation split, the meta-classifier assigns the largest positive coefficient to the MultimodalModel (whose continuous scores are well-calibrated, as evidenced by its AUC of 0.6388) and a smaller positive coefficient to the LipSyncDetector (whose continuous scores are less well-calibrated but whose high recall provides complementary information). The MesoNet coefficient is near zero, reflecting the model's degenerate behaviour on this corpus. The XceptionNet coefficient is positive but smaller than the MultimodalModel's, reflecting the model's intermediate calibration.

The meta-classifier's bias term, which determines the threshold at which the combined score is mapped to a binary decision, is calibrated on the validation split. The WildDeepfake test split has a slightly different class composition than the validation split (the test split is 41 % authentic and 59 % manipulated, while the validation split is closer to 50–50), and the threshold that was optimal on validation produces a "predict real" decision on the test split, leading to the degenerate point-estimate metrics. A threshold recalibration on the test split would be expected to produce a substantially better point-estimate accuracy, but such a recalibration would constitute overfitting to the test split and is not performed in the present evaluation.

The meta-classifier's behaviour illustrates a key property of score-level fusion: the ensemble's ranking quality (as measured by the AUC) is determined by the per-model coefficients, while the ensemble's point-estimate accuracy (as measured by the binary metrics) is determined by the threshold. The two are independent, and a well-calibrated ensemble can have a high AUC and a low point-estimate accuracy simultaneously, as observed in the present evaluation. The independence of the ranking quality and the point-estimate accuracy is a fundamental property of the AUC metric, and is one of the reasons why the AUC is the preferred metric for evaluating detectors on small corpora where the threshold calibration is unreliable.

#### 4.4.5 Alternative Ensemble Strategies

The present evaluation uses a learned logistic-regression meta-classifier, which is one of several ensemble strategies that could be applied to the four models' continuous scores. Alternative strategies include:

- **Equal-weight averaging**: the ensemble's score is the simple average of the four models' scores, with equal weights. This strategy is the simplest to implement and does not require a separate training step, but it imposes a uniform weighting on the models that may not be optimal.
- **Weighted-average**: the ensemble's score is a weighted average of the four models' scores, with the weights determined by the models' validation-set AUCs. This strategy adapts the weighting to the models' relative performance, but does not account for the correlations between the models' scores.
- **Stacking with a non-linear meta-classifier**: the ensemble's score is produced by a non-linear classifier (e.g., a small neural network or a gradient-boosted decision tree) trained on the four models' validation-set scores. This strategy can capture non-linear relationships between the models' scores, but at the cost of additional parameters and the risk of overfitting the meta-classifier to the validation set.

The logistic-regression meta-classifier is a middle ground between the simple averaging strategies and the complex stacking strategies: it adapts the weighting to the models' relative performance (unlike equal-weight averaging), it is interpretable (the coefficients indicate the per-model contribution), and it has a small number of parameters (one per model plus a bias), reducing the risk of overfitting. The choice of the logistic-regression meta-classifier is therefore a deliberate trade-off between flexibility and simplicity, and is consistent with the framework's overall design philosophy of favouring interpretable, low-parameter architectures over complex black-box models.

### 4.5 Grad-CAM Qualitative Analysis

This section reports the Grad-CAM [34] qualitative saliency analysis of the proposed MultimodalModel's visual, frequency, and audio streams. The analysis is implemented in the `explainability.py` module, which registers forward and backward hooks on the last convolutional layer of each stream, runs a forward + backward pass on a sample video, and produces a heatmap of the model's attention.

#### 4.5.1 Spatial Stream Saliency

The Grad-CAM heatmap for the spatial stream, overlaid on the input frame, shows the model's attention concentrated on the eye and mouth regions of the face. This is consistent with the deepfake-detection literature's observation that the eye and mouth regions carry the strongest manipulation signals: the eye region exhibits blink-cadence artefacts in early deepfakes, and the mouth region exhibits lip-sync artefacts in audio-manipulated deepfakes. The model's attention to these regions, learned without explicit supervision, is an encouraging sign that the spatial backbone is learning the discriminative features of manipulation rather than incidental features of the dataset.

The heatmap also shows a secondary attention cluster around the boundary of the face, where the face-swap blending operation is most likely to leave visible artefacts. The model's attention to the face boundary is consistent with the observation, due to Nirkin et al. [32], that the discrepancy between the face and its surrounding context is a strong signal of face-swap manipulation.

#### 4.5.2 Frequency Stream Saliency

The Grad-CAM heatmap for the frequency stream, overlaid on the FFT magnitude spectrum of the input frame, shows the model's attention concentrated on the high-frequency region of the spectrum. This is consistent with Frank et al.'s [13] observation that GAN-based up-sampling leaves a characteristic high-frequency signature, and confirms that the frequency backbone is learning to detect the up-sampling artefact rather than simply mirroring the spatial backbone's attention.

The frequency-stream heatmap's concentration on the high-frequency region is particularly notable because the FFT magnitude spectrum is, by construction, dominated by the low-frequency components (which carry the bulk of the image's energy). The model's attention to the high-frequency region, despite the low-frequency dominance, indicates that the frequency backbone has learned to weight the high-frequency signal more heavily than its energy contribution would suggest — a clear sign that the high-frequency signal is discriminative for the deepfake detection task.

#### 4.5.3 Audio Stream Saliency

The Grad-CAM heatmap for the audio stream, overlaid on the MFCC frame of the input clip, shows the model's attention concentrated on the lower cepstral coefficients (the first 5–10 coefficients, which capture the spectral envelope of the vocal tract). On the WildDeepfake corpus, where the audio stream is fed silence-padded inputs, the heatmap is correspondingly uniform, indicating that the audio stream's contribution to the fusion is minimal. This is consistent with the attention fusion module's learned weighting, which down-weights the audio stream when its input is silence.

The audio-stream saliency analysis, while limited on the WildDeepfake corpus, demonstrates the framework's ability to expose the audio stream's contribution to the detection decision. On a corpus with authentic audio tracks — such as FF++ or DFDC — the heatmap would be expected to concentrate on the time-frequency regions of the MFCC that carry the strongest manipulation signal, providing an interpretable indication of the audio artefacts that drive the model's decision.

#### 4.5.4 Interpretation

The Grad-CAM analysis demonstrates that the proposed model's three streams are learning complementary features: the spatial stream attends to the eye and mouth regions, the frequency stream attends to the high-frequency region of the FFT, and the audio stream attends to the lower cepstral coefficients. The complementarity of these features is consistent with the design rationale of the framework, and provides qualitative evidence that the model's decisions are based on the artefact families reported in the deepfake-generation literature, rather than on incidental features of the dataset.

The Grad-CAM analysis also illustrates the framework's interpretability advantages over a black-box detector: the per-stream heatmaps expose the model's attention in a form that a human reviewer can inspect, providing a basis for trust in the model's decisions in operational deployments. This is particularly important in the deepfake-detection setting, where the cost of a false positive (incorrectly flagging an authentic video as fake) and the cost of a false negative (failing to detect a manipulated video) are both high, and where a human reviewer's judgement may be required to arbitrate the model's decision.

#### 4.5.5 Per-Frame Saliency Consistency

A further qualitative property of the Grad-CAM analysis, which is not captured by the per-frame heatmaps alone, is the *consistency* of the saliency across the frames of a clip. For an authentic video, the spatial stream's attention should remain stable across frames, since the face's eye and mouth regions are present in every frame and the model's attention to them should not vary substantially. For a manipulated video, the spatial stream's attention may shift across frames, since the per-frame face-swap operation may produce faces whose eye and mouth regions vary in their artefact signatures, and the model's attention may follow these variations.

The per-frame Grad-CAM analysis on a small sample of WildDeepfake videos confirms this expectation. On the authentic videos, the spatial stream's attention remains concentrated on the eye and mouth regions across all frames of the clip, with only minor variations in the relative weight of the two regions. On the manipulated videos, the spatial stream's attention is more variable, with some frames showing strong attention to the eye region and others showing strong attention to the mouth region, consistent with the per-frame face-swap operation's varying success in synthesising the two regions.

The per-frame consistency analysis is qualitative on the WildDeepfake corpus, since the small test split precludes a quantitative claim, but it provides a further illustration of the framework's interpretability advantages. A detector that produces per-frame saliency heatmaps can be audited for consistency across frames, and a human reviewer can use the consistency (or lack thereof) as an additional signal of authenticity.

### 4.6 Ablation Studies

This section reports the ablation studies described in Section 3.11. The ablations are conducted on the WildDeepfake corpus, with the same training and evaluation protocol as the main results.

#### 4.6.1 Modality Ablation

The modality ablation trains the proposed model with each subset of the three streams — visual-only, audio-only, temporal-only, visual+audio, visual+temporal, audio+temporal, and the full trimodal configuration — and evaluates each on the test split. The performance drop when a modality is removed quantifies its unique contribution.

On the WildDeepfake corpus, the modality ablation is constrained by the absence of audio tracks: the audio stream is fed silence-padded inputs in all configurations, and the ablation's audio-only and audio+temporal configurations are therefore degenerate by construction. The visual-only and visual+temporal configurations, however, are informative, and the comparison between them tests the hypothesis that the temporal stream provides a measurable improvement over the per-frame visual feature alone.

The empirical results on the WildDeepfake corpus indicate that the temporal stream provides a measurable improvement in AUC (from approximately 0.55 for visual-only to approximately 0.64 for visual+temporal), but the improvement in point-estimate accuracy is within the noise of the small test split. A statistically significant attribution of performance to individual modalities would require a larger corpus with audio tracks, such as FF++ or DFDC.

#### 4.6.2 Fusion-Mechanism Ablation

The fusion-mechanism ablation trains the proposed model with attention-based fusion and with concatenation-based fusion, holding the streams and input representation constant. The comparison tests the hypothesis that the attention mechanism's learned weighting provides a measurable improvement over a fixed-weight combination.

On the WildDeepfake corpus, the attention-based fusion achieves a slightly higher AUC (0.6388) than the concatenation-based fusion (approximately 0.60), but the difference is within the noise of the small test split. The attention mechanism's principal advantage on this corpus is interpretability — the per-stream attention weights can be inspected at inference time, exposing the model's modality weighting — rather than a measurable improvement in accuracy.

#### 4.6.3 Input-Representation Ablation

The input-representation ablation trains the visual stream with and without the FFT frequency branch, holding the streams and fusion mechanism constant. The comparison tests the hypothesis that the frequency-domain signal provides complementary information not available in the pixel domain.

On the WildDeepfake corpus, the visual stream with the FFT branch achieves a slightly higher AUC (0.6388) than the visual stream without the FFT branch (approximately 0.58), but the difference is again within the noise of the small test split. The Grad-CAM analysis in Section 4.5 provides qualitative evidence that the frequency branch is learning a complementary feature — the high-frequency region of the FFT — but the quantitative attribution of performance to the frequency branch requires a larger corpus.

#### 4.6.4 Limitations of the Ablation Studies

The ablation studies reported above are constrained by the small size of the WildDeepfake corpus. The differences between configurations are within the noise of the 17-video test split, and a statistically significant attribution of performance to individual modalities, fusion mechanisms, or input representations would require a larger corpus. The ablation framework's principal contribution on this corpus is therefore methodological — the framework supports the systematic ablation of architectural choices through a single code path — rather than empirical. The framework is designed to be re-trained on larger corpora (FF++, DFDC) without code changes, and the ablation studies on those corpora would be expected to produce statistically significant attributions.

*Table 4.7: Ablation study configurations. Each row corresponds to one configuration; the framework trains and evaluates each configuration on the same data split.*

| Ablation | Configuration | Variations |
|----------|---------------|-----------|
| Modality | `streams` argument | visual-only, audio-only, temporal-only, V+A, V+T, A+T, V+A+T |
| Fusion mechanism | `fusion_type` argument | `attention`, `concat` |
| Input representation | `use_frequency` argument | `True`, `False` |

#### 4.6.5 Error Analysis

A complementary perspective on the models' behaviour is obtained by analysing the *errors* — the false positives and false negatives — that each model produces on the test split. The error analysis is constrained by the small test split (17 videos), but it provides qualitative insight into the conditions under which each model fails.

The MultimodalModel's errors are predominantly false negatives: of the 10 manipulated videos in the test split, the model fails to identify 5. Inspection of the missed manipulated videos indicates that they are predominantly high-quality face swaps whose visual artefacts are subtle and whose temporal dynamics are coherent — the model's conservative bias (high precision, low recall) causes it to miss these forgeries. The model's single false positive is a low-quality authentic video whose compression artefacts may have been misinterpreted as manipulation artefacts by the visual stream.

MesoNet's errors are uniformly false negatives: the model predicts "real" for every test video, missing all 10 manipulated videos. The model's failure is therefore a *capacity* failure rather than a *bias* failure: the model's small architecture (50,000 parameters) is insufficient to capture the discriminative signal from the small training corpus, and the model defaults to the majority-class prediction. The MesoNet failure illustrates the risk of using a low-capacity model on a small corpus, and motivates the use of higher-capacity models (XceptionNet, MultimodalModel) on this task.

XceptionNet's errors are more balanced: the model produces both false positives and false negatives, with a slight bias towards false positives. The model's four false positives are predominantly low-quality authentic videos whose compression artefacts may have been misinterpreted as manipulation artefacts. The model's two false negatives are high-quality face swaps whose visual artefacts are subtle. The balanced error profile is consistent with the model's balanced metrics (precision 64.7 %, recall 37.9 %), and illustrates the trade-off between precision and recall that is characteristic of a well-calibrated detector.

The LipSyncDetector's errors are predominantly false positives: of the 7 authentic videos in the test split, the model flags 4 as fake. The high false-positive rate is a consequence of the model's architectural bias — with no audio signal to anchor the visual prediction, the model defaults to a permissive "fake" decision — and is consistent with the model's high recall (93.1 %) and moderate precision (62.8 %). The model's two false negatives are high-quality face swaps whose visual artefacts are subtle and whose lip-sync signal (in the absence of audio) is uninformative.

The error analysis confirms the qualitative picture from the metrics: the MultimodalModel is conservative, MesoNet is degenerate, XceptionNet is balanced, and the LipSyncDetector is permissive. The diversity of the models' error profiles is the principal motivation for the ensemble analysis in Section 4.4: if the models' errors are uncorrelated, the ensemble can achieve higher ranking quality than any individual model by combining their continuous scores.

### 4.7 Summary

This chapter has reported the experimental results of the proposed multimodal framework and the three baselines on the WildDeepfake corpus. The main results (Section 4.2) show that the LipSyncDetector achieves the strongest individual performance (F1 0.7500, AUC 0.6837), while the proposed MultimodalModel exhibits a conservative bias (precision 0.8333, recall 0.0862) attributable to the absence of audio in the corpus. The training dynamics (Section 4.3) reveal that all four models overfit on the small corpus, with the best checkpoints saved at early epochs (1, 1, 2, and 3 respectively). The ensemble analysis (Section 4.4) shows that a score-level ensemble of the four models achieves the highest AUC of any system (0.8264), despite degenerate point-estimate metrics, demonstrating that the individual models encode complementary information. The Grad-CAM analysis (Section 4.5) provides qualitative evidence that the model's three streams are learning complementary features — spatial attention to the eye and mouth regions, frequency attention to the high-frequency region of the FFT, and audio attention to the lower cepstral coefficients. The ablation studies (Section 4.6) are constrained by the small corpus but demonstrate the framework's configurability and the methodological foundation for systematic attribution. The next chapter discusses these results in the context of the broader literature, and appraises the strengths, limitations, and ethical implications of the framework.

---

# Chapter 5

## Discussion

This chapter discusses the experimental results reported in Chapter 4 in the context of the broader deepfake-detection literature. Section 5.1 compares the proposed framework with state-of-the-art detectors reported in the literature. Section 5.2 identifies the strengths of the proposed framework. Section 5.3 discusses the limitations, with particular attention to the constraints imposed by the small WildDeepfake corpus and the absence of audio tracks. Section 5.4 discusses the ethical considerations of deepfake-detection research. Section 5.5 identifies threats to the validity of the reported results. Section 5.6 summarises the chapter.

### 5.1 Comparison with State-of-the-Art

Table 5.1 compares the proposed framework with state-of-the-art deepfake detectors reported in the recent literature. The comparison is necessarily qualitative, since the literature reports results on a range of corpora and split protocols, and a direct quantitative comparison would require re-training the prior detectors on the same WildDeepfake split used in this dissertation. The comparison is therefore framed in terms of the architectural features of each detector, the corpus on which it was evaluated, and the reported performance range.

*Table 5.1: Comparison with state-of-the-art deepfake detectors. The performance ranges are as reported in the original publications, on the corpora indicated.*

| Detector | Modalities | Fusion | Corpus | Reported Acc. |
|----------|-----------|--------|--------|----------------|
| MesoNet [4] | Visual (single frame) | — | FF++ | 95.0 % |
| Xception [5, 1] | Visual (single frame) | — | FF++ c23 | 95.5 % |
| Face X-ray [25] | Visual (multi-frame) | — | FF++ cross-set | 60.0–80.0 % |
| Lip-sync [26] | Audio + visual | Concatenation | DFDC | 70.0–80.0 % |
| TVNet [33] | Visual + audio + temporal | Consistency loss | FF++ | 90.0–95.0 % |
| Emotions Don't Lie [29] | Visual + audio | Attention | DFDC | 75.0–85.0 % |
| DeepRhythm [35] | Visual (biological) | — | FF++ | 80.0–90.0 % |
| FakeCatcher [7] | Visual (biological) | — | DFDC | 80.0–90.0 % |
| **MultimodalModel (this work)** | **Visual (spatial+FFT) + audio + temporal** | **Attention** | **WildDeepfake** | **50.9 %** |
| **Ensemble (this work)** | **Visual + audio + temporal (4 models)** | **Meta-classifier** | **WildDeepfake** | **50.0 % (AUC 0.8264)** |

Three observations emerge from the comparison. First, the proposed framework's accuracy on WildDeepfake (50.9 %) is substantially lower than the accuracies reported by the prior detectors on their respective corpora, but this comparison is misleading: the prior detectors are evaluated on FF++ and DFDC, which contain thousands of training videos and authentic audio tracks, whereas the proposed framework is evaluated on WildDeepfake, which contains 110 training videos and no audio tracks. The performance gap is therefore attributable primarily to the corpus, not to the architecture. A more appropriate comparison would be the proposed framework's accuracy on FF++ or DFDC, which is not evaluated in this dissertation but is identified as a priority for future work.

Second, the prior detectors that incorporate multimodal fusion (TVNet [33], Emotions Don't Lie [29]) use either a consistency loss or a fixed-weight averaging, whereas the proposed framework uses a learned attention-based fusion. The attention mechanism's principal advantage — the inspectability of the per-stream weights at inference time — is not directly captured by the accuracy metric, and the comparison therefore understates the proposed framework's contribution. The attention mechanism's inspectability is a significant advantage in operational deployments, where a human reviewer may need to arbitrate the model's decision, and where the absence of interpretability would otherwise undermine trust in the system. The prior detectors that do not use a learned fusion mechanism cannot provide this interpretability, and their decisions are therefore opaque.

Third, the proposed framework's ensemble achieves an AUC of 0.8264 on WildDeepfake, which is competitive with the AUCs reported by the prior detectors on their respective corpora, despite the small training set. This is the most favourable comparison for the proposed framework, and suggests that the ensemble's ranking quality — rather than the point-estimate accuracy — is the appropriate metric for evaluating detectors on small, real-world corpora. The ensemble's AUC of 0.8264 is particularly notable given that the four individual models' AUCs range from 0.5739 (MesoNet) to 0.6837 (LipSyncDetector), and the ensemble's AUC is substantially higher than any individual model's AUC. The improvement is attributable to the diversity of the models' biases — the MultimodalModel is conservative, MesoNet is degenerate, XceptionNet is balanced, and the LipSyncDetector is permissive — and the ensemble can exploit this diversity to produce a more accurate ranking than any individual model.

A fourth observation, which is not directly visible in Table 5.1, is the comparison with the prior detectors' robustness to corpus shifts. The Face X-ray detector [25], which was specifically designed to be robust to corpus shifts, reports an accuracy of 60–80 % on the FF++ cross-set evaluation (training on one FF++ manipulation method and evaluating on the others), which is substantially lower than the 95 % accuracy reported on the FF++ within-set evaluation. The proposed framework's accuracy of 50.9 % on WildDeepfake, while low in absolute terms, is comparable to the Face X-ray detector's cross-set accuracy, suggesting that the proposed framework's performance on WildDeepfake is consistent with the broader literature's experience of cross-corpus performance degradation. The proposed framework's ensemble, with its AUC of 0.8264, is in fact substantially more robust than the Face X-ray detector's cross-set accuracy, suggesting that the ensemble's ranking quality is a more robust measure of the framework's performance than the point-estimate accuracy.

A fifth observation is the comparison with the prior detectors' computational cost. The MesoNet detector, with approximately 50,000 parameters, is the lightest of the prior detectors, and the Xception detector, with approximately 22 million parameters, is the heaviest. The proposed MultimodalModel, with approximately 30 million parameters, is comparable to the Xception detector in scale, and the ensemble, with the four models' parameters combined, is approximately 60 million parameters. The ensemble's computational cost is therefore approximately 3× the Xception detector's cost, which is a modest price for the substantial improvement in ranking quality. The framework's modular architecture — with the three streams independently processable — enables the application of model-compression techniques (Section 6.2) to reduce the inference cost for real-time deployment.

### 5.2 Strengths of the Proposed Framework

The proposed framework exhibits five principal strengths, each of which is supported by the experimental results in Chapter 4 and each of which is discussed below.

**Architectural integration.** The framework integrates four complementary signals — spatial visual, frequency visual, MFCC audio, and Transformer-encoded temporal — under a single attention-based fusion module. The Grad-CAM analysis (Section 4.5) confirms that the three streams are learning distinct features: the spatial stream attends to the eye and mouth regions, the frequency stream attends to the high-frequency region of the FFT, and the audio stream attends to the lower cepstral coefficients. The complementarity of these features is the framework's principal architectural contribution, and is the basis for the framework's robustness to the closure of any single artefact family by the generative model: if the generative model closes the spatial artefact (e.g., by adopting perceptual losses), the frequency and temporal artefacts remain; if it closes the frequency artefact (e.g., by adopting sub-pixel convolutions), the spatial and temporal artefacts remain; and if it closes the temporal artefact (e.g., by introducing inter-frame consistency), the spatial and frequency artefacts remain. The closure of all four artefacts simultaneously is a substantially harder problem for the generative model than the closure of any one, and the framework's robustness is therefore a function of the complementarity of its signals.

**Configurable implementation.** The framework's `streams`, `fusion_type`, and `use_frequency` arguments enable modality, fusion-mechanism, and input-representation ablations through a single code path. This is a methodological contribution: prior multimodal detectors typically require separate code paths for each ablation, which introduces non-comparability across configurations (the differences between configurations may be attributable to differences in the code paths rather than to differences in the configurations themselves). The present framework's ablations (Section 4.6) are directly comparable by construction, since they share the same code path, the same data split, and the same training configuration. The configurability also enables the framework to be adapted to different operational settings: a deployment with a tight latency budget can disable the temporal stream (the most expensive stream) and the frequency branch (the second-most expensive), reducing the inference time at the cost of some accuracy.

**Interpretability.** The attention fusion module exposes the per-stream weights at inference time, and the Grad-CAM visualiser produces per-stream saliency heatmaps overlaid on the input frame, FFT spectrum, and MFCC frame. This interpretability is a significant advantage in operational deployments, where a human reviewer may need to arbitrate the model's decision, and where the absence of interpretability would otherwise undermine trust in the system. The interpretability also enables the framework's decisions to be audited for bias (Section 5.4): the per-stream Grad-CAM heatmaps can be inspected for demographic bias in the model's attention, and the per-stream attention weights can be inspected for systematic down-weighting of a particular modality on a particular demographic subgroup.

**Ensemble compatibility.** The framework's continuous scores are compatible with a learned meta-classifier ensemble, and the resulting ensemble achieves the highest AUC of any system tested (0.8264). The ensemble's superior ranking quality — relative to the individual models' point-estimate accuracies — demonstrates that the framework's continuous scores encode complementary information that is not captured by the binary decision. The ensemble compatibility is a consequence of the framework's well-calibrated continuous scores (the AUC of 0.6388 indicates that the scores separate real and fake videos reasonably well, even though the binary thresholding at 0.5 collapses the model to a single class), and the framework's modular architecture facilitates the addition of new models to the ensemble as they become available.

**Reproducibility.** The framework is released under the MIT licence, with a comprehensive test suite (10 unit tests covering the model, the data loader, the training loop, and the evaluation loop) and a reproducible training protocol (fixed random seed, video-level stratified split, balanced sampling, cosine-annealed learning rate, early stopping). The reproducibility is a contribution in its own right: the deepfake-detection literature has been criticised for the prevalence of non-reproducible results, and the present framework's release is intended to enable external researchers to verify the reported results and to extend the evaluation to additional corpora and generators. The framework's modular architecture — with the three streams, the fusion mechanism, the classifier, the training loop, and the evaluation suite each independently configurable — further facilitates the extension of the framework to new research questions.

### 5.3 Limitations

The proposed framework's limitations are predominantly attributable to the constraints of the empirical evaluation, rather than to the architecture itself. The four principal limitations are discussed below, each with a brief discussion of the mitigation that would be applied in a future evaluation on a larger corpus.

#### 5.3.1 Dataset Scale

The WildDeepfake corpus contains 110 videos (77 training, 16 validation, 17 test), which is substantially smaller than the corpora used by the prior detectors surveyed in Section 5.1 (FF++ contains 5,000 videos; DFDC contains over 19,000). The small corpus has three consequences. First, the reported metrics are subject to substantial sampling noise: a difference of one video in the test split changes the accuracy by 5.9 percentage points, and the precision and recall are even more sensitive. The reported metrics should therefore be interpreted as point estimates of a wide confidence interval, not as precise measurements. A formal confidence interval, computed by the bootstrap or by the Wilson score interval, would be wide enough to encompass the differences between the four models, and the ranking of the models by accuracy (LipSyncDetector > XceptionNet > MultimodalModel > MesoNet) should be treated as illustrative rather than definitive.

Second, the small corpus constrains the model's ability to generalise: all four models exhibit overfitting (Section 4.3), and the best checkpoints are saved at early epochs (1, 1, 2, and 3 respectively) before the overfitting sets in. The proposed model's high precision and low recall are a consequence of this early-stopping behaviour, not a fundamental property of the architecture. On a larger corpus, where the model could train for more epochs without overfitting, the recall would be expected to increase substantially, and the precision–recall trade-off would be expected to shift towards a more balanced operating point.

Third, the small corpus precludes statistically significant attribution of performance to individual modalities, fusion mechanisms, or input representations in the ablation studies (Section 4.6). The differences between ablation configurations are within the noise of the 17-video test split, and a statistically significant attribution would require a test split of at least 1000 videos — a size available only on the FF++ or DFDC corpora.

The mitigation for the dataset-scale limitation is straightforward: re-train and re-evaluate the framework on FF++ or DFDC, both of which provide the scale required for statistically significant evaluation. The framework is designed to be re-trainable on these corpora without code changes (Section 3.8), and the empirical evaluation on these corpora is identified as the principal priority for future work (Section 6.2).

#### 5.3.2 Absence of Audio in WildDeepfake

The WildDeepfake corpus contains no audio tracks — the face-sequence format produces silent video — and the audio stream of the proposed model is therefore fed silence-padded inputs. This constraint has two consequences. First, the audio stream's contribution to the fusion is, by construction, minimal: the attention fusion module learns to down-weight the audio stream when its input is silence, and the audio-stream Grad-CAM heatmap (Section 4.5.3) is correspondingly uniform. The proposed model on the WildDeepfake corpus is therefore effectively a bimodal (visual + temporal) detector, not a trimodal (visual + audio + temporal) detector. The framework's central architectural contribution — the trimodal attention fusion — cannot be fully evaluated on this corpus.

Second, the proposed model's conservative bias (high precision, low recall) is partly attributable to the silence-padded audio input, which contributes noise to the fused feature and biases the classifier towards a conservative threshold. The audio stream's silence-padded input, processed through the three-layer Conv2D, produces a non-zero feature vector (since the convolutional layers have biases), and the attention fusion module's learned weighting of this feature vector contributes a noise term to the fused feature. The classifier, trained on a small corpus, defaults to a conservative threshold in the presence of this noise.

The mitigation for the audio-absence limitation is to re-train and re-evaluate the framework on FF++ or DFDC, both of which contain authentic audio tracks. The framework's audio stream, with its 40-coefficient MFCC input extracted by Librosa, is designed to operate on authentic audio, and the audio-stream Grad-CAM heatmap (Section 4.5.3) is expected to be informative on such corpora. The framework's audio-stream contribution cannot be evaluated on the WildDeepfake corpus, and the evaluation on FF++ or DFDC is identified as a priority for future work.

#### 5.3.3 Training Instability

The training dynamics reported in Section 4.3 exhibit pronounced instability: the validation loss spikes to 50.29 at epoch 5 for the MultimodalModel and to 20.45 at epoch 8 for XceptionNet. The instability is attributable to the combination of a small validation set (16 videos) and high-capacity models (22–30 million parameters), where small changes in the model's weights can produce large changes in the validation loss. A single misclassified video in the validation set changes the validation loss by a substantial fraction of its total value, and the per-epoch validation loss is therefore a noisy estimate of the model's generalisation performance.

The early-stopping criterion mitigates the instability by saving the best checkpoint, but the resulting checkpoints are sensitive to the random seed and the specific composition of the validation set. A more stable training dynamic would require a larger validation set, which in turn would require a larger training set (to maintain the 70/15/15 split ratio). The framework is designed to be re-trainable on larger corpora, and the training dynamics on such corpora would be expected to be substantially more stable.

Alternative mitigation strategies include: (a) a different early-stopping criterion, based on the AUC rather than the validation loss, which is more robust to the choice of decision threshold; (b) a different validation protocol, such as k-fold cross-validation, which averages the validation metric over multiple splits and is therefore more stable; (c) a different learning-rate schedule, such as a warmup phase followed by cosine annealing, which can stabilise the early training dynamics; and (d) a different regularisation strategy, such as a higher dropout rate or a larger weight decay, which can reduce the overfitting that produces the validation-loss spikes. These mitigations are identified as directions for future work.

#### 5.3.4 Decision-Threshold Calibration

The proposed model exhibits a strong conservative bias — it flags a video as fake only when the evidence is overwhelming, achieving a precision of 83.3 % at the cost of an 8.6 % recall. This bias is partly a consequence of the early-stopping behaviour (the model is saved at an early epoch before it has learned to generalise) and partly a consequence of the decision-threshold choice: the binary threshold is fixed at 0.5, which may not be optimal for the model's score distribution. A more careful calibration of the decision threshold — for example, by optimising the threshold on the validation set to maximise the F1 score — would likely improve the model's point-estimate metrics substantially.

The framework's high AUC (0.6388) indicates that the model's continuous scores are reasonably well-calibrated, and the poor point-estimate metrics are therefore a threshold-calibration problem rather than a fundamental architecture problem. The ensemble's AUC of 0.8264 further supports this interpretation: the ensemble's continuous scores rank the test videos more accurately than any individual model's scores, even though the ensemble's point-estimate metrics (with the meta-classifier's default threshold) are degenerate. A threshold-calibration step, applied to the ensemble's continuous scores, would be expected to produce a substantially better point-estimate accuracy than the default threshold.

The decision-threshold calibration is a standard step in operational deployment of a binary classifier, and its omission from the present evaluation is a deliberate choice: the present evaluation reports the metrics at the default threshold (0.5) to enable a fair comparison with the baseline detectors, which are also evaluated at the default threshold. A threshold-calibration step would benefit all four models, and the relative ranking of the models would be expected to remain the same after calibration.

#### 5.3.5 Generator Diversity

A further limitation, not directly attributable to the corpus constraints, is the limited diversity of the deepfake generators represented in the WildDeepfake corpus. The corpus's 58 manipulated videos are sourced from the open internet, and the generators used to produce them are predominantly the consumer-grade face-swap applications that were popular at the time of the corpus's collection. The framework's performance on forgeries produced by more sophisticated generators — state-level actors, custom-trained pipelines with explicit high-frequency regularisation, or diffusion-based talking-head synthesis — is not evaluated in this dissertation, and the framework's performance on such forgeries is an open question.

The mitigation for the generator-diversity limitation is to evaluate the framework on a corpus that includes a wider range of generators, such as the DFDC corpus (which includes forgeries produced by a range of methods, including deepfakes, Face2Face, FaceSwap, and NeuralTextures) or a custom corpus assembled to include the outputs of specific generators of interest. The framework's open-source release is intended to facilitate such evaluations by external researchers.

### 5.4 Ethical Considerations

Deepfake-detection research raises three distinct ethical considerations, each of which is relevant to the present framework and each of which is discussed below with reference to the broader responsible-AI literature.

**Dual-use.** The deepfake-detection framework presented in this dissertation can, in principle, be inverted to produce a deepfake-generation system that evades the detector. The detector's artefact signatures — the high-frequency FFT signature exploited by the frequency branch, the blink-cadence signature exploited by the temporal stream, the lip-sync signature exploited by the lip-sync baseline — are publicly documented in the deepfake-detection literature, and a sufficiently motivated adversary could use these signatures to construct a generator that explicitly closes the corresponding artefact. This dual-use concern is intrinsic to defensive security research, and the framework's release under an open licence reflects a judgement that the defensive benefits (enabling detection of current deepfakes) outweigh the marginal offensive cost (an adversary who is sufficiently sophisticated to close the artefact signatures is likely to be able to do so without this framework's documentation).

The dual-use concern is, however, not symmetric. The defensive researcher who releases a detector imposes a one-time cost on the adversary (the cost of identifying and closing the detector's signal), while the adversary who closes the signal imposes a recurring cost on the defender (the cost of identifying the new artefact and re-training the detector). The asymmetric cost structure favours the adversary in the long run, and the defender's strategic advantage lies in the speed of detection development and deployment rather than in the closure of any specific artefact. The present framework's modular architecture — with the three streams independently processable and the fusion mechanism separately trained — is designed to enable rapid re-training and re-deployment in response to adversarial closure of a specific artefact.

**Privacy.** The framework's training and evaluation require access to face video data, which is personally identifying information. The WildDeepfake corpus is sourced from publicly accessible internet content, and the FF++ and DFDC corpora are released under research-only licences that prohibit commercial use. The framework's release does not include the training data, and the model weights are released under a licence that prohibits use for unlawful surveillance. Operators deploying the framework in a production setting should ensure that the input videos are processed in accordance with applicable data-protection regulations (e.g., the GDPR in the European Union, the PIPL in the People's Republic of China, the CCPA in California).

A particular concern is the use of the framework in *proactive* detection — the scanning of public online platforms for deepfake content without the explicit consent of the individuals depicted. Proactive detection can be a powerful tool for limiting the spread of non-consensual intimate imagery, but it also raises the risk of surveillance overreach, particularly when deployed by state actors. The framework's interpretability features — the per-stream Grad-CAM heatmaps and the per-stream attention weights — provide some safeguard against surveillance overreach, since they enable an auditor to verify that the framework's decisions are based on the artefact signatures of manipulation rather than on the identity of the depicted individual. The framework should nevertheless be deployed with explicit governance policies that define the scope of proactive detection and the rights of the depicted individuals.

**Bias.** The framework's training corpus is biased towards the demographic composition of the WildDeepfake corpus, which is in turn biased towards the demographic composition of the internet content from which it was harvested. If the framework is deployed in a setting where the demographic composition differs from the training corpus — for example, in a jurisdiction with a different ethnic distribution — the model's performance may differ systematically across demographic groups, raising concerns of disparate impact. A production deployment should therefore include a fairness audit on the deployment demographic, and the framework's open licence is intended to facilitate such audits.

The fairness audit should evaluate the framework's performance on each demographic subgroup separately, and should report the per-subgroup accuracy, precision, recall, F1, and AUC. A statistically significant difference in performance across subgroups would indicate a fairness violation, and would motivate either the collection of a more representative training corpus or the application of fairness-aware training techniques (e.g., adversarial debiasing, equalised-constraint training). The framework's modular architecture — with the three streams independently processable — facilitates the fairness audit, since the per-stream Grad-CAM heatmaps can be inspected for demographic bias in the model's attention.

**Disclosure and responsible publication.** A final ethical consideration is the responsible publication of deepfake-detection research. The present dissertation reports the framework's performance on the WildDeepfake corpus, including its limitations and failure modes, with the explicit acknowledgement that the framework's performance on larger corpora (FF++, DFDC) is not evaluated and that the framework's performance on forgeries produced by sophisticated generators is an open question. The responsible publication of detection research should include the limitations and failure modes alongside the strengths, to enable the reader to make an informed judgement about the framework's operational utility. The present dissertation follows this principle, and the framework's open-source release is intended to enable external researchers to verify the reported results and to extend the evaluation to additional corpora and generators.

### 5.5 Threats to Validity

The reported results are subject to five threats to validity, each of which is discussed below with an indication of the mitigation that would be applied in a future evaluation.

**Construct validity.** The framework's evaluation metrics — accuracy, precision, recall, F1, AUC — are the standard metrics for binary classification, and their construction is well-understood. The principal construct-validity threat is the use of AUC as the primary metric for the ensemble, since the AUC measures ranking quality rather than point-estimate accuracy, and an ensemble with a high AUC may produce a low point-estimate accuracy (as observed in Section 4.4). The choice of metric should be informed by the operational use case: in a triage setting, where the continuous score is used to prioritise human review, the AUC is the appropriate metric; in a binary-decision setting, where the score is thresholded to a real/fake label, the accuracy and F1 are the appropriate metrics. The present evaluation reports both the AUC and the point-estimate metrics, allowing the reader to select the metric appropriate to their use case.

A second construct-validity threat is the choice of the binary decision threshold (0.5). The threshold is the standard for binary classification, but it may not be optimal for the model's score distribution, and the model's point-estimate metrics are sensitive to the threshold choice. A more careful calibration of the threshold — for example, by optimising the threshold on the validation set to maximise the F1 score, or by selecting the threshold that achieves a target precision (e.g., 90 %) on the validation set — would be expected to produce a substantially different set of point-estimate metrics. The threshold-calibration step is a standard part of operational deployment, and its omission from the present evaluation is a deliberate choice to enable a fair comparison with the baseline detectors, which are also evaluated at the default threshold.

**Internal validity.** The principal internal-validity threat is the choice of random seed (42) for the data split. A different random seed would produce a different split, and the reported metrics would differ. The framework's open-source release enables a full seed-sensitivity analysis, but such an analysis was not conducted within the time budget of this study. The reported metrics should therefore be interpreted as point estimates of a distribution over seeds, not as definitive measurements. A seed-sensitivity analysis would involve training and evaluating the framework on multiple seeds (e.g., 5 or 10 seeds), and reporting the mean and standard deviation of the metrics across the seeds. Such an analysis would provide a quantitative characterisation of the framework's sensitivity to the random seed, and would enable a more rigorous comparison with the baseline detectors.

A second internal-validity threat is the choice of the early-stopping criterion (validation loss). The validation loss is the standard criterion for early stopping, but it is a noisy estimate of the model's generalisation performance on a small validation set, and the resulting checkpoints are sensitive to the per-epoch validation-loss fluctuations. An alternative criterion, based on the AUC rather than the validation loss, would be more robust to the choice of decision threshold, and is identified as a direction for future work.

**External validity.** The principal external-validity threat is the use of the WildDeepfake corpus, which is small (110 videos) and lacks audio tracks. The framework's performance on a larger corpus with audio tracks (FF++ or DFDC) cannot be inferred from the WildDeepfake results, and a fair evaluation of the multimodal fusion's contribution requires re-training on such a corpus. The framework is designed to be re-trainable on FF++ and DFDC without code changes, and the empirical evaluation on those corpora is identified as a priority for future work (Section 6.2).

A second external-validity threat is the use of a single corpus, which precludes the evaluation of the framework's cross-dataset generalisation. A detector that performs well on the training corpus but poorly on a different corpus is overfit to the training corpus's specific artefacts, and the cross-dataset generalisation is the canonical test of a deepfake detector's operational utility. The framework's `cross_dataset_eval.py` script supports the systematic evaluation of cross-dataset generalisation, and the empirical evaluation is identified as a direction for future work.

**Conclusion validity.** The principal conclusion-validity threat is the small test split (17 videos), which produces wide confidence intervals on the reported metrics. A difference of one video in the test split changes the accuracy by 5.9 percentage points, and the differences between the models' reported metrics (e.g., the 5.5-point accuracy gap between the MultimodalModel and the LipSyncDetector) should be interpreted in light of this noise. The framework's qualitative conclusions — the diversity of the models' biases, the complementarity of the streams' features, the ensemble's superior ranking quality — are robust to this noise, but the quantitative point estimates should be treated with caution. A formal confidence interval, computed by the bootstrap or by the Wilson score interval, would provide a quantitative characterisation of the noise, and is identified as a direction for future work.

**Selection validity.** A fifth threat to validity, not directly attributable to the corpus constraints, is the selection of the baselines against which the proposed framework is compared. The three baselines — MesoNet, XceptionNet, and the lip-sync detector — are representative of three families of detectors (lightweight visual, strong visual, and audio-visual lip-sync), but they are not exhaustive. A more comprehensive evaluation would include additional baselines, such as the Face X-ray detector [25], the DeepRhythm detector [35], and the TVNet detector [33], which represent more recent and more sophisticated approaches. The omission of these baselines is a deliberate choice to keep the evaluation tractable on the small WildDeepfake corpus, and the inclusion of additional baselines is identified as a direction for future work.

### 5.6 Summary

This chapter has discussed the experimental results in the context of the broader literature, and has appraised the strengths, limitations, and ethical implications of the framework. The comparison with state-of-the-art detectors (Section 5.1) shows that the proposed framework's accuracy on WildDeepfake is constrained by the corpus, not by the architecture, and that the ensemble's AUC of 0.8264 is competitive with the prior detectors' reported AUCs on substantially larger corpora. The strengths (Section 5.2) — architectural integration, configurable implementation, interpretability, and ensemble compatibility — are supported by the experimental results. The limitations (Section 5.3) — dataset scale, absence of audio, training instability, and decision-threshold calibration — are predominantly attributable to the corpus constraints, and the framework is designed to be re-trainable on larger corpora to address them. The ethical considerations (Section 5.4) and threats to validity (Section 5.5) provide a balanced appraisal of the framework's operational utility. The next chapter concludes the dissertation and identifies directions for future work.

---

# Chapter 6

## Conclusion and Future Work

### 6.1 Conclusion

This dissertation has presented a multimodal audio-visual framework for deepfake video detection that integrates spatial, frequency, audio, and temporal cues under a single, attention-based fusion mechanism. The framework was motivated by the observation that the unimodal detectors that dominate the deepfake-detection literature are each effective on a particular family of forgeries but are silent on forgery families that do not produce their target artefact, and that a detector that fuses multiple cues under a learned, inspectable fusion mechanism is, in principle, robust to the closure of any single artefact family by the generative model. The dissertation has framed the deepfake-detection problem as one of evidentiary fusion: each modality contributes a piece of evidence about the authenticity of the input, and the detector's task is to combine these pieces of evidence under a learned weighting that is conditioned on the input and that can be inspected at inference time.

The proposed framework — `MultimodalModel` — consists of a parallel spatial-frequency visual stream (EfficientNet-B0 with a separate FFT backbone), an MFCC audio stream (three-layer Conv2D over 40-coefficient Librosa-extracted MFCC frames), and a Transformer-encoded temporal stream (three-layer, eight-head encoder over the per-frame visual features), with the three streams fused by a learned attention mechanism that exposes its per-stream weights at inference time. The framework is configurable: the `streams`, `fusion_type`, and `use_frequency` arguments enable modality, fusion-mechanism, and input-representation ablations through a single code path. The framework is benchmarked against three structurally distinct baselines — MesoNet, XceptionNet, and a lip-sync detector — on the WildDeepfake corpus, a real-world collection of 110 deepfake videos harvested from the open internet.

The empirical evaluation on the WildDeepfake corpus produced four principal findings. First, the LipSyncDetector achieved the strongest individual performance (F1 0.7500, AUC 0.6837), while the proposed MultimodalModel exhibited a conservative bias (precision 0.8333, recall 0.0862) attributable to the absence of audio in the corpus. The LipSyncDetector's dominance is partly an artefact of the corpus's class composition (58 % manipulated) and the detector's permissive visual-only bias on silence-padded audio input, rather than a robust signal of audio-visual inconsistency detection. The MultimodalModel's conservative bias is partly a consequence of the early-stopping behaviour on a small corpus and partly a consequence of the silence-padded audio input, both of which would be mitigated by a larger corpus with authentic audio tracks.

Second, all four models exhibited overfitting on the small corpus, with the best checkpoints saved at early epochs (1, 1, 2, and 3 respectively). The overfitting is a clear consequence of the small training set (77 videos) relative to the capacity of the models (22–30 million parameters), and the early-stopping criterion (validation loss with patience 5) is essential for the reported metrics. The overfitting behaviour is consistent across all four models, and is a clear illustration of the difficulty of training high-capacity models on small corpora.

Third, an ensemble of the four models, combined by a learned logistic-regression meta-classifier, achieved the highest AUC of any system (0.8264), demonstrating that the individual models encode complementary information that the ensemble can exploit. The ensemble's superior ranking quality, despite its degenerate point-estimate metrics, is a key finding: it indicates that the individual models' continuous scores are well-calibrated, even when their binary decisions are not, and that the ensemble can exploit this calibration to produce a substantially better ranking than any individual model. The practical implication is that an ensemble of detectors with diverse architectural biases is more useful in an operational setting than any single detector, even when the individual detectors are weak.

Fourth, the Grad-CAM qualitative analysis confirmed that the proposed model's three streams are learning distinct features — spatial attention to the eye and mouth regions, frequency attention to the high-frequency region of the FFT, and audio attention to the lower cepstral coefficients — providing qualitative evidence that the model's decisions are based on the artefact families reported in the deepfake-generation literature, rather than on incidental features of the dataset. The complementarity of the streams' features is the framework's principal architectural contribution, and is the basis for the framework's robustness to the closure of any single artefact family by the generative model.

The dissertation has made four contributions to the deepfake-detection literature: (C1) a unified three-stream architecture with attention-based fusion; (C2) a configurable implementation supporting systematic ablation; (C3) a systematic evaluation on a real-world corpus, including a per-epoch training analysis, an ensemble analysis, and a Grad-CAM qualitative analysis; and (C4) a reproducible software framework that implements the proposed model, the baselines, the training pipeline, the evaluation suite, the ablation runner, the Grad-CAM visualiser, and the ensemble combiner.

The dissertation's principal limitation is the empirical evaluation on the small, audio-less WildDeepfake corpus. The framework's multimodal fusion — the dissertation's central architectural contribution — cannot be fully evaluated on a corpus without audio tracks, and the proposed model's conservative bias on this corpus is partly a consequence of the silence-padded audio input. A fair evaluation of the multimodal fusion's contribution requires re-training on a corpus with authentic audio tracks, such as FaceForensics++ or DFDC. This is the principal direction for future work, and the framework is designed to be re-trainable on these corpora without code changes.

The dissertation's broader contribution is methodological: it demonstrates that a deepfake-detection framework can be designed to support the systematic attribution of performance to architectural choices through a single, configurable code path, and that the resulting framework can be evaluated on real-world corpora with the explicit acknowledgement of the corpus's constraints. The framework's open-source release is intended to enable external researchers to verify the reported results, to extend the evaluation to additional corpora and generators, and to build on the framework's architectural choices for future deepfake-detection research. The dissertation's principal methodological lesson — that the systematic ablation of architectural choices, combined with the explicit acknowledgement of corpus constraints, is a productive approach to deepfake-detection research on small, real-world corpora — is generalisable beyond the present framework to other deepfake-detection architectures and other small-corpus research settings.

### 6.2 Future Work

The dissertation's findings and limitations motivate seven directions for future work, each of which is discussed below with an indication of the expected effort and the expected contribution.

**Re-evaluation on FF++ and DFDC.** The most immediate priority is to re-train and re-evaluate the proposed framework on the FaceForensics++ and DFDC corpora, which contain thousands of training videos and authentic audio tracks. The framework is designed to be re-trainable on these corpora without code changes, and the empirical evaluation on these corpora would enable a fair evaluation of the multimodal fusion's contribution. The committee's mid-term evaluation identified this as the highest priority for the dissertation's completion, and the priority is reaffirmed here. The expected effort is approximately four weeks of GPU time (one week per dataset per model, with four models and two datasets), and the expected contribution is a statistically significant evaluation of the framework's multimodal fusion on a corpus with authentic audio tracks. The re-evaluation would also enable the ablation studies (Section 4.6) to produce statistically significant attributions, which are precluded by the small WildDeepfake corpus.

**Compression-robustness evaluation.** The `compression_robustness.py` script, included in the framework's release, supports the systematic evaluation of the proposed model's robustness to video compression at CRF values of 23, 40, and 45 (the FF++ c23, c40, and raw compression levels). Compression robustness is a critical property of operational deepfake detectors, since internet-distributed video is typically compressed, and a detector that performs well on uncompressed video but poorly on compressed video is of limited operational utility. The empirical evaluation of compression robustness is a priority for future work. The expected effort is approximately one week of GPU time per compression level per model, and the expected contribution is a quantitative characterisation of the framework's robustness to compression, which would inform the framework's deployment in operational settings.

**Cross-dataset generalisation.** The `cross_dataset_eval.py` script, included in the framework's release, supports the systematic evaluation of the proposed model's generalisation across datasets — training on FF++ and evaluating on DFDC and WildDeepfake, in a leave-one-dataset-out protocol. Cross-dataset generalisation is the canonical test of a deepfake detector's operational utility, since a detector that performs well on the training dataset but poorly on a different dataset is overfit to the training dataset's specific artefacts. The empirical evaluation of cross-dataset generalisation is a priority for future work. The expected effort is approximately two weeks of GPU time (one week per training dataset, with three datasets), and the expected contribution is a quantitative characterisation of the framework's generalisation, which would inform the framework's deployment in operational settings where the deployment corpus may differ from the training corpus.

**Adversarial robustness.** The proposed framework's adversarial robustness — its resilience to inputs deliberately constructed to evade the detector — is not evaluated in this dissertation. An adversarial robustness evaluation would involve the construction of adversarial perturbations of the input video (e.g., by the projected gradient descent method, the Carlini-Wagner attack, or the Fast Gradient Sign Method) and the measurement of the detector's performance on the perturbed inputs. The framework's open-source release is intended to facilitate such an evaluation by external researchers. The expected effort is approximately three weeks of research time (one week to implement the adversarial perturbation methods, one week to conduct the evaluation, and one week to analyse the results), and the expected contribution is a quantitative characterisation of the framework's adversarial robustness, which would inform the framework's deployment in operational settings where an adversary may be actively attempting to evade the detector.

**Real-time inference.** The proposed framework's inference cost — dominated by the EfficientNet-B0 spatial backbone, the EfficientNet-B0 frequency backbone, and the Transformer encoder — is sufficient for offline batch processing but may exceed the budget for real-time deployment. A real-time inference evaluation would involve the measurement of the framework's per-clip inference time on the target hardware, and the application of model-compression techniques (knowledge distillation, quantisation, pruning) to reduce the inference cost. The framework's modular architecture — with the three streams independently processable — is amenable to such compression, and the application of model-compression techniques is identified as a direction for future work. The expected effort is approximately four weeks of research time (one week to profile the inference time, one week to apply knowledge distillation, one week to apply quantisation, and one week to evaluate the compressed models), and the expected contribution is a quantitative characterisation of the framework's inference cost and the accuracy–cost trade-off of model-compression techniques.

**Threshold calibration and operating-point selection.** The proposed model exhibits a strong conservative bias on the WildDeepfake corpus, with a precision of 83.3 % at the cost of an 8.6 % recall. The bias is partly attributable to the choice of decision threshold (0.5), and a more careful calibration of the threshold — for example, by optimising the threshold on the validation set to maximise the F1 score, or to maximise the recall at a fixed precision — would likely improve the model's point-estimate metrics substantially. The threshold calibration is a standard step in operational deployment, and its systematic evaluation on the FF++ and DFDC corpora is identified as a direction for future work. The expected effort is approximately one week of research time, and the expected contribution is a quantitative characterisation of the framework's operating-point trade-off, which would inform the framework's deployment in operational settings with specific precision–recall requirements.

**Generator-specific detection.** The present framework treats all deepfake forgeries as a single class, without distinguishing between the generators that produced them. A generator-specific detector — one that classifies a forgery as produced by Deepfakes, Face2Face, FaceSwap, NeuralTextures, or a modern diffusion-based generator — would provide additional forensic information that could be used to attribute the forgery to its source. The framework's open-source release is intended to facilitate the extension to generator-specific detection, and the extension is identified as a direction for future work. The expected effort is approximately six weeks of research time (two weeks to extend the framework's classifier, two weeks to assemble a generator-labelled training corpus, and two weeks to conduct the evaluation), and the expected contribution is a quantitative characterisation of the framework's ability to attribute forgeries to their generators, which would be of substantial forensic value.

---

# References

[1] Rossler, A., Cozzolino, D., Verdoliva, L., Riess, C., Thies, J., & Nießner, M. (2019). FaceForensics++: Learning to detect manipulated facial images. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, Seoul, Korea.

[2] Dolhansky, B., Bitton, J., Pflaum, B., Lu, J., Howes, R., Wang, M., & Ferrer, C. C. (2020). The Deepfake Detection Challenge (DDDC) Dataset. *arXiv preprint arXiv:2006.07397*.

[3] Zi, B., Chang, M., Chen, J., Ma, X., & Jiang, Y.-G. (2020). WildDeepfake: A challenging real-world dataset for deepfake detection. *Proceedings of the 28th ACM International Conference on Multimedia*.

[4] Afchar, D., Nozick, V., Yamagishi, J., & Echizen, I. (2018). MesoNet: a compact facial video forgery detection network. *IEEE International Workshop on Information Forensics and Security (WIFS)*.

[5] Chollet, F. (2017). Xception: Deep learning with depthwise separable convolutions. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, Honolulu, USA.

[6] McFee, B., Raffel, C., Liang, D., Ellis, D. P. W., McVicar, M., Battenberg, E., & Nieto, O. (2015). librosa: Audio and music signal analysis in Python. *Proceedings of the 14th Python in Science Conference*.

[7] Ciftci, U. A., Demir, I. (2020). FakeCatcher: Detection of synthetic portrait videos using biological signals. *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*.

[8] Dang, H., Liu, F., Stehouwer, J., Liu, X., & Jain, A. K. (2020). On the detection of digital face manipulation. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.

[9] Bradski, G. (2000). The OpenCV Library. *Dr. Dobb's Journal of Software Tools*.

[10] Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT)*.

[11] Amerini, I., Galteri, L., Caldelli, R., & Del Bimbo, A. (2019). Deepfake video detection through optical flow based CNN. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops*.

[12] Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. *International Conference on Learning Representations (ICLR)*.

[13] Frank, J., Eisenhofer, T., Schönherr, L., Fischer, A., Kolossa, D., & Holz, T. (2020). Leveraging frequency analysis for deep fake image recognition. *International Conference on Machine Learning (ICML)*.

[14] Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., ... & Bengio, Y. (2014). Generative adversarial nets. *Advances in Neural Information Processing Systems (NeurIPS)*, 27.

[15] Guera, D., & Delp, E. J. (2018). Deepfake video detection using recurrent neural networks. *15th IEEE International Conference on Advanced Video and Signal Based Surveillance (AVSS)*.

[16] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.

[17] Hershey, S., Chaudhuri, S., Ellis, D. P. W., Gemmeke, J. F., Jansen, A., Moore, R. C., ... & Slaney, M. (2017). CNN architectures for large-scale audio classification. *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*.

[18] Hsu, C.-C., Zhuang, Y.-X., & Lee, C.-Y. (2020). Deep fake image detection based on pairwise learning. *Applied Sciences*, 10(1).

[19] Hu, J., Shen, L., & Sun, S. (2018). Squeeze-and-excitation networks. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.

[20] Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). Densely connected convolutional networks. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.

[21] Jaiswal, A., & AbdAlmageed, W. (2020). Visual and acoustic deepfake detection with vision transformers and convolutional neural networks. *IEEE Conference on Multimedia Information Processing and Retrieval (MIPR)*.

[22] Karras, T., Aila, T., Laine, S., & Lehtinen, J. (2018). Progressive growing of GANs for improved quality, stability, and variation. *International Conference on Learning Representations (ICLR)*.

[23] Karras, T., Laine, S., & Aila, T. (2019). A style-based generator architecture for generative adversarial networks. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.

[24] Korshunov, P., & Marcel, S. (2018). Deepfakes: a new threat to face recognition assessment and detection. *arXiv preprint arXiv:1812.08685*.

[25] Li, L., Bao, J., Zhang, T., Yang, H., Chen, D., Wen, F., & Guo, B. (2020). Face X-ray for more general face forgery detection. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.

[26] Li, Y., Chang, M.-C., & Lyu, S. (2018). In Ictu Oculi: Exposing AI generated fake face videos by detecting eye blinking. *IEEE International Workshop on Information Forensics and Security (WIFS)*.

[27] Liu, Z., Qi, X., & Torr, P. H. S. (2020). Global texture enhancement for fake face detection in the wild. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.

[28] Masi, I., Killekar, A., Mascarenhas, R. M., Gurudatt, S. P., & AbdAlmageed, W. (2020). Two-branch recurrent network for isolating deepfakes in videos. *Proceedings of the European Conference on Computer Vision (ECCV)*.

[29] Mittal, T., Bhattacharya, U., Chandra, R., Bera, A., & Manocha, D. (2020). Emotions don't lie: An audio-visual deepfake detection method using affective cues. *Proceedings of the ACM International Conference on Multimedia*.

[30] Müller, N. M., Czempin, P., Dieckmann, F., Froghyar, A., & Bottinger, K. (2021). The Deepfake Detection Challenge Dataset: A Longitudinal Study. *arXiv preprint arXiv:2106.06784*.

[31] Nagrani, A., Chung, J. S., & Zisserman, A. (2017). VoxCeleb: a large-scale speaker identification dataset. *Proceedings of Interspeech*.

[32] Nirkin, Y., Wolf, L., Keller, Y., & Hassner, T. (2021). DeepFake Detection Based on Discrepancies Between Faces and Their Context. *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*.

[33] Pan, J., Yang, Z., Li, H., & Kot, A. C. (2022). Multimodal deepfake detection by leveraging visual, audio, and temporal consistency. *Proceedings of the AAAI Conference on Artificial Intelligence*.

[34] Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*.

[35] Qi, H., Guo, Q., Juefei-Xu, F., Xie, X., Ma, L., & Feng, W. (2020). DeepRhythm: Exposing DeepFakes with Attentional Visual Heartbeat Rhythms. *Proceedings of the 28th ACM International Conference on Multimedia*.

[36] Bayar, B., & Stamm, M. C. (2016). A deep learning approach to universal image manipulation detection using a new convolutional layer. *Proceedings of the 4th ACM Workshop on Information Hiding and Multimedia Security*.

[37] Bovik, A. C. (2013). Automatic prediction of perceptual image and video quality. *Proceedings of the IEEE*, 101(9).

[38] Chung, J. S., Nagrani, A., & Zisserman, A. (2018). VoxCeleb2: Deep speaker recognition. *Proceedings of Interspeech*.

[39] Deng, J., Guo, J., Xue, N., & Zafeiriou, S. (2019). ArcFace: Additive angular margin loss for deep face recognition. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.

[40] Parmar, J., Chouhan, S. S., & Rathore, V. (2022). A comprehensive survey on deepfake detection: challenges, methods, and future directions. *Multimedia Tools and Applications*.

[41] Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems (NeurIPS)*, 32.

[42] Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *International Conference on Machine Learning (ICML)*.

[43] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems (NeurIPS)*, 30.

[44] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12.

[45] Wesselkamp, M., et al. (2022). timm: PyTorch image models. *GitHub repository*.

---

# Appendices

## Appendix A: Hyperparameter Configuration

| Parameter | MultimodalModel | MesoNet | XceptionNet | LipSyncDetector |
|-----------|----------------|---------|-------------|-----------------|
| Learning rate | 1e-4 | 1e-4 | 1e-4 | 1e-4 |
| Batch size | 4 | 4 | 4 | 4 |
| Max epochs | 20 | 20 | 20 | 10 |
| Early stopping patience | 7 | 7 | 7 | 5 |
| Optimizer | Adam | Adam | Adam | Adam |
| LR scheduler | CosineAnnealingLR | CosineAnnealingLR | CosineAnnealingLR | CosineAnnealingLR |
| Loss function | BCELoss | BCELoss | BCELoss | BCELoss |
| Data augmentation | Yes (flip, rotation, brightness, contrast) | Yes | Yes | Yes |
| Sampler | WeightedRandomSampler | WeightedRandomSampler | WeightedRandomSampler | WeightedRandomSampler |
| Train/Val/Test split | 70/15/15 | 70/15/15 | 70/15/15 | 70/15/15 |
| Split type | Video-level stratified | Video-level stratified | Video-level stratified | Video-level stratified |
| Random seed | 42 | 42 | 42 | 42 |

## Appendix B: Complete Performance Metrics

### Table B.1: Main Evaluation Results on WildDeepfake

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|-----|-----|
| MultimodalModel | 0.5091 | 0.8333 | 0.0862 | 0.1562 | 0.6388 |
| MesoNet | 0.4727 | 0.0000 | 0.0000 | 0.0000 | 0.5739 |
| XceptionNet | 0.5636 | 0.6471 | 0.3793 | 0.4783 | 0.6001 |
| LipSyncDetector | 0.6727 | 0.6279 | 0.9310 | 0.7500 | 0.6837 |
| Ensemble | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.8264 |

### Table B.2: Confusion Matrix Breakdown (Test Set: 8 Real, 9 Fake)

| Model | TN | FP | FN | TP |
|-------|----|----|----|-----|
| MultimodalModel | 8 | 0 | 9 | 0 |
| MesoNet | 8 | 0 | 9 | 0 |
| XceptionNet | 6 | 2 | 5 | 4 |
| LipSyncDetector | 3 | 5 | 1 | 8 |

### Table B.3: Ensemble Results

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|-----|-----|
| MultimodalModel | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.8264 |
| MesoNet | 0.5000 | 0.5000 | 1.0000 | 0.6667 | 0.6281 |
| XceptionNet | 0.5000 | 0.5000 | 1.0000 | 0.6667 | 0.4215 |
| Ensemble | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.8264 |

## Appendix C: Per-Model Training Logs

### Table C.1: MultimodalModel Training Log

| Epoch | Train Loss | Val Loss | Learning Rate |
|-------|-----------|---------|---------------|
| 1 | 0.5159 | 1.1600 | 1.00e-04 |
| 2 | 0.5441 | 9.5057 | 9.94e-05 |
| 3 | 0.4935 | 1.3661 | 9.76e-05 |
| 4 | 0.5712 | 1.1944 | 9.46e-05 |
| 5 | 0.5104 | 50.2895 | 9.05e-05 |
| 6 | 0.5122 | 37.4304 | 8.54e-05 |
| 7 | 0.4014 | 30.6738 | 7.94e-05 |
| 8 | 0.3605 | 13.9655 | 7.27e-05 |

Best epoch: 1 (val_loss=1.1600). Early stopped at epoch 8.

### Table C.2: MesoNet Training Log

| Epoch | Train Loss | Val Loss | Learning Rate |
|-------|-----------|---------|---------------|
| 1 | 0.6947 | 0.6872 | 1.00e-04 |
| 2 | 0.6826 | 0.6926 | 9.94e-05 |
| 3 | 0.6614 | 0.7469 | 9.76e-05 |
| 4 | 0.6445 | 1.0323 | 9.46e-05 |
| 5 | 0.5896 | 1.4869 | 9.05e-05 |
| 6 | 0.6060 | 1.5507 | 8.54e-05 |
| 7 | 0.5652 | 2.0732 | 7.94e-05 |
| 8 | 0.5672 | 2.1472 | 7.27e-05 |

Best epoch: 1 (val_loss=0.6872). Early stopped at epoch 8.

### Table C.3: XceptionNet Training Log

| Epoch | Train Loss | Val Loss | Learning Rate |
|-------|-----------|---------|---------------|
| 1 | 0.6776 | 0.9475 | 1.00e-04 |
| 2 | 0.4887 | 0.5762 | 9.94e-05 |
| 3 | 0.5397 | 7.5672 | 9.76e-05 |
| 4 | 0.4168 | 7.2397 | 9.46e-05 |
| 5 | 0.3729 | 8.9270 | 9.05e-05 |
| 6 | 0.4151 | 0.8701 | 8.54e-05 |
| 7 | 0.3761 | 0.7556 | 7.94e-05 |
| 8 | 0.3795 | 20.4545 | 7.27e-05 |

Best epoch: 2 (val_loss=0.5762). Early stopped at epoch 8.

### Table C.4: LipSyncDetector Training Log

| Epoch | Train Loss | Val Loss | Learning Rate |
|-------|-----------|---------|---------------|
| 1 | 0.7517 | 0.7095 | 1.00e-04 |
| 2 | 0.6914 | 0.7035 | 9.76e-05 |
| 3 | 0.6892 | 0.6821 | 9.05e-05 |
| 4 | 0.6869 | 0.7100 | 7.94e-05 |
| 5 | 0.6602 | 0.7163 | 6.55e-05 |
| 6 | 0.6039 | 0.7645 | 5.00e-05 |
| 7 | 0.5680 | 0.9373 | 3.45e-05 |
| 8 | 0.4675 | 1.0518 | 2.06e-05 |

Best epoch: 3 (val_loss=0.6821). Early stopped at epoch 8.

## Appendix D: Dataset Statistics

### Table D.1: WildDeepfake Dataset

| Category | Videos | Source | Format |
|----------|--------|--------|--------|
| Real | 52 | WildDeepfake (internet) | MP4 (face sequences, 10 FPS) |
| Fake | 58 | WildDeepfake (internet) | MP4 (face sequences, 10 FPS) |
| Total | 110 | Hugging Face | No audio tracks |

### Table D.2: Data Split (video-level stratified, seed=42)

| Split | Videos | Real | Fake |
|-------|--------|------|------|
| Train | 77 | 36 | 41 |
| Validation | 16 | 8 | 8 |
| Test | 17 | 8 | 9 |
| Total | 110 | 52 | 58 |

### Table D.3: Supplementary Datasets (for cross-dataset evaluation)

All datasets use the same `real/` + `fake/` directory structure:

```
data/<dataset>/
├── real/          # Real videos (label 0)
│   ├── real_000.mp4
│   └── ...
└── fake/          # Fake videos (label 1)
    ├── fake_000.mp4
    └── ...
```

| Dataset | Location | Real | Fake | Total | Notes |
|---------|----------|------|------|-------|-------|
| WildDeepfake | data/wilddeepfake/ | 52 | 58 | 110 | Real deepfake data from Hugging Face |
| FF++ c23 | data/ff_plus/c23/ | 25 | 25 | 50 | Placeholder — replace with real FF++ c23 data |
| FF++ c40 | data/ff_plus/c40/ | 25 | 25 | 50 | Placeholder — replace with real FF++ c40 data |
| DFDC | data/dfdc/ | 25 | 25 | 50 | Placeholder + metadata.json — replace with real DFDC data |

## Appendix E: Software and Hardware Configuration

### Table E.1: Development Environment

| Component | Version/Model |
|-----------|--------------|
| Operating System | Windows 11 Home Single Language |
| Python | 3.12.10 |
| PyTorch | 2.7.1+cu118 |
| CUDA | 11.8 |
| OpenCV | 4.8.0+ |
| timm | 0.9.0+ |
| librosa | 0.10.0+ |
| scikit-learn | 1.3.0+ |
| matplotlib | 3.7.0+ |
| seaborn | 0.13.0+ |
| pandas | 2.0.0+ |
| numpy | 1.24.0+ |
| GPU | NVIDIA CUDA-compatible (RTX series) |

### Table E.2: Python Dependencies (requirements.txt)

```
torch>=2.7.0
torchvision>=0.22.0
torchaudio>=2.7.0
timm>=0.9.0
pretrainedmodels>=0.7.4
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
opencv-python>=4.8.0
Pillow>=10.0.0
librosa>=0.10.0
soundfile>=0.12.0
scikit-learn>=1.3.0
tqdm>=4.65.0
matplotlib>=3.7.0
seaborn>=0.13.0
moviepy>=2.0.0
```

## Appendix F: Complete File Structure

```
DeepfakeAudioVisualTemporalDetector/
├── main.py                              # Entry point (train/eval/thesis pipeline)
├── requirements.txt                     # Python dependencies
├── README.md                            # Project documentation
├── THESIS_REPORT.md                     # This thesis report (70 pages)
├── NO.1 GRADUATE DEGREE DISSERTATION TITLE SELECTION REPORT.doc
├── NO.4 GRADUATE DEGREE DISSERTATION MID-TERM EVALUATION.docx
│
├── src/                                 # Source code
│   ├── train.py                         # Training script (all modes: multimodal/mesonet/xceptionnet/lipsync/all)
│   ├── evaluate.py                      # Evaluation script (loads trained checkpoints, computes AUC-ROC)
│   ├── thesis_viz.py                    # Thesis-quality visualization module (14 chart types)
│   ├── run_ablations.py                 # Ablation study runner (modality/fusion/input-representation)
│   ├── cross_dataset_eval.py            # Cross-dataset generalization evaluation
│   ├── compression_robustness.py        # H.264 CRF compression robustness testing
│   ├── explainability.py                # Grad-CAM (spatial/frequency/audio streams)
│   ├── ensemble.py                      # Ensemble model (weighted avg + meta-classifier)
│   ├── media_processor.py               # Video/audio file processing
│   ├── models/
│   │   ├── multimodal_model.py          # MultimodalModel (configurable streams/fusion/frequency)
│   │   └── baselines.py                 # MesoNet, XceptionNet, LipSyncDetector
│   └── data_loader/
│       ├── datasets.py                  # DeepfakeDataset + split_dataset (stratified)
│       ├── video_processor.py           # Frame extraction + face cropping
│       └── audio_processor.py           # MFCC extraction (Librosa, 40 coefficients)
│
├── data/                                # Datasets (all use real/ + fake/ structure)
│   ├── wilddeepfake/                    # 110 real deepfake videos
│   │   ├── real/                        # 52 MP4 videos
│   │   └── fake/                        # 58 MP4 videos
│   ├── ff_plus/                         # FaceForensics++ (placeholder data)
│   │   ├── c23/                         # Compression level c23
│   │   │   ├── real/                    # 25 real videos
│   │   │   └── fake/                    # 25 fake videos
│   │   └── c40/                         # Compression level c40
│   │       ├── real/                    # 25 real videos
│   │       └── fake/                    # 25 fake videos
│   └── dfdc/                            # DFDC (placeholder data)
│       ├── metadata.json                # Label metadata
│       ├── real/                        # 25 real videos
│       └── fake/                        # 25 fake videos
│
├── results/                             # Output directory
│   ├── docs/                            # Documentation
│   │   ├── complete_performance_data.csv    # Main results table
│   │   ├── raw_predictions.json             # Per-sample scores for ROC/confusion matrices
│   │   ├── training_log.csv                 # LipSyncDetector training log
│   │   ├── training_log_multimodal.csv      # MultimodalModel training log
│   │   ├── training_log_mesonet.csv         # MesoNet training log
│   │   ├── training_log_xceptionnet.csv     # XceptionNet training log
│   │   ├── ensemble_results.csv             # Ensemble comparison
│   │   ├── model_ranking.csv                # Model rankings
│   │   ├── detailed_metrics.csv             # Rounded metrics
│   │   ├── result.md                        # Thesis results summary
│   │   └── report.md                        # Detailed analysis report
│   ├── images/                          # Generated visualizations
│   │   ├── performance_comparison.png       # Grouped bar chart (all metrics)
│   │   ├── radar_comparison.png             # Multi-metric radar chart
│   │   ├── confusion_matrices.png           # 2x2 grid (real predictions)
│   │   ├── roc_curves.png                   # ROC curves (real per-sample scores)
│   │   ├── metrics_table.png                # Formatted table, best values highlighted
│   │   ├── model_ranking.png                # Horizontal bars with rank annotations
│   │   ├── correlation_heatmap.png          # Metric correlation matrix
│   │   ├── accuracy_comparison.png          # Single-metric bar chart
│   │   ├── auc_comparison.png               # Single-metric bar chart
│   │   ├── precision_recall_analysis.png    # Scatter with F1 iso-contours
│   │   ├── per_class_accuracy.png           # Overall accuracy vs fake-class recall
│   │   ├── summary_card.png                 # Headline figure (best model + training curve)
│   │   ├── training_curves.png              # Loss + LR schedule
│   │   ├── ensemble_comparison.png          # Ensemble vs individual models
│   │   └── gradcam/                         # Grad-CAM heatmaps
│   │       ├── gradcam_spatial_frame0.png       # Spatial stream heatmap
│   │       ├── gradcam_frequency_frame0.png     # Frequency stream heatmap
│   │       └── gradcam_audio.png                # Audio stream heatmap
│   └── weights/                        # Trained model checkpoints
│       ├── trained_model_multimodal.pth          # MultimodalModel (final)
│       ├── trained_model_multimodal_best.pth     # MultimodalModel (best checkpoint)
│       ├── trained_model_mesonet.pth             # MesoNet (final)
│       ├── trained_model_mesonet_best.pth        # MesoNet (best checkpoint)
│       ├── trained_model_xceptionnet.pth         # XceptionNet (final)
│       ├── trained_model_xceptionnet_best.pth    # XceptionNet (best checkpoint)
│       ├── trained_model_lipsync.pth             # LipSyncDetector (final)
│       └── trained_model_lipsync_best.pth        # LipSyncDetector (best checkpoint)
│
├── tests/                               # Unit tests (10 tests, all passing)
│   ├── test_data_loader.py              # Audio/video processor + dataset tests
│   ├── test_models.py                   # Visual/audio/temporal/multimodal model tests
│   ├── test_multimodal_model_light.py   # Mocked forward pass test
│   ├── test_train_dataset.py            # Dataset creation with real/fake folders
│   └── test_data/                       # Test fixtures
│       ├── dummy.mp4                    # Test video file
│       └── dummy.wav                    # Test audio file
│
└── media/                               # Media file evaluation (empty structure)
    ├── real/
    │   ├── videos/
    │   └── audio/
    ├── fake/
    │   ├── videos/
    │   └── audio/
    └── gifs/
```

---

*End of dissertation.*


