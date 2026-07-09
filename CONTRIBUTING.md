# Contributing to DeepfakeAudioVisualTemporalDetector

Thank you for your interest in contributing! This is a research project for a Master's thesis, but extensions and improvements are welcome.

## How to Contribute

### Reporting Bugs
1. Check existing [issues](https://github.com/theraihanrakibb/DeepfakeAudioVisualTemporalDetector/issues) to avoid duplicates
2. Open a new issue with the **Bug Report** template
3. Include: Python version, PyTorch version, OS, GPU model, error traceback

### Suggesting Features
1. Open a new issue with the **Feature Request** template
2. Describe the use case and expected behavior

### Submitting Code
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Run tests: `pytest tests/ -v` (all 10 must pass)
5. Commit with a clear message
6. Open a Pull Request using the PR template

### Code Style
- Follow existing Python style (PEP 8)
- Keep functions focused and documented
- Add tests for new functionality
- Don't break existing tests

### Areas for Contribution
- **FF++ integration**: Test with real FaceForensics++ data
- **DFDC integration**: Test with real DFDC data
- **Compression robustness**: Requires FFmpeg — test with real CRF values
- **Threshold calibration**: Implement learned or temperature-scaled thresholds
- **Meta-classifier ensemble**: Train logistic regression on validation predictions
- **Real-time inference**: Optimize for video streaming scenarios
