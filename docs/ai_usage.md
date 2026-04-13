## Usage Log

### 1. Code Generation: Data Loader Implementation

**Tool:** Claude / ChatGPT-4 / GitHub Copilot

**Prompt Used:**
```
"Write a PyTorch DataLoader for CIFAR-10 with proper normalization using mean [0.4914, 0.4822, 0.4465] 
and std [0.2023, 0.1994, 0.2010]. Include support for both training and test splits with appropriate 
data augmentation."
```

**How It Was Modified:**
- [x] Added proper error handling for missing directories
- [x] Customized normalization to exactly match paper's preprocessing
- [x] Added docstrings and type hints
- [x] Tested on multiple data splits
- [x] Verified output shapes match expected CIFAR-10 format

**Verification:** Tested with actual CIFAR-10 data; shapes and values verified correct.

---

### 2. Code Generation: LRP Attribution Method

**Tool:** Claude

**Prompt Used:**
```
"Create a PyTorch function to compute Layer-wise Relevance Propagation (LRP) attributions using Zennit. 
The function should:
1. Accept a model and input tensor
2. Support EpsilonPlus, EpsilonAlpha2Beta1, and SumRule composite types
3. Return attributions with same shape as input
4. Include proper error handling"
```

**How It Was Modified:**
- [x] Added support for target class specification
- [x] Implemented attribution aggregation functions
- [x] Added normalization utility
- [x] Tested compatibility with paper's specifications
- [x] Added comprehensive docstrings

**Verification:** Tested with ResNet50 on CIFAR-10; compared shapes with paper's reported values.

---

### 3. Documentation & Comments

**Tool:** Claude

**Prompt Used:**
```
"Write detailed docstrings for LRP attribution functions following NumPy documentation style. 
Include Args, Returns, Raises, and Examples sections."
```

**How It Was Used:**
- Adapted generated docstrings to our actual implementation
- Customized examples to match our project context
- Added parameter descriptions specific to our use case

---

### 4. Debugging: Attribution Shape Mismatch

**Tool:** GitHub Copilot


**Issue:** "ValueError: Attribution shape (1, 3, 32, 32) doesn't match input (1, 10)"

**AI Suggestion:**
```python
# Check if attributions are being incorrectly squeezed
# Restore full shape before aggregation
```

**How Issue Was Resolved:**
- [x] Identified shape mismatch in attribution aggregation
- [x] Verified fix with multiple input batches
- [x] Added assertion checks in code
- [x] Tested with all three XAI methods

**Verification:** Successfully runs without shape errors; outputs match expected dimensions.

---

## Declaration

I declare that:

- ✅ **All AI usage documented above** - Every significant use of AI tools is logged
- ✅ **All prompts recorded** - Original prompts are included verbatim
- ✅ **All outputs verified** - AI-generated code was tested and validated
- ✅ **Code modified and understood** - I understand all generated code and made improvements
- ✅ **No direct copying** - AI outputs were adapted and customized for our project
- ✅ **Proper attribution** - AI usage is clearly marked in comments where applicable

---

## Code Comments for AI Usage

Where AI tools generated code, comments have been added:

```python
# NOTE: Initial implementation generated with assistance from Claude
# Modified to add error handling and custom error messages
def compute_attribution(...):
    ...
```


### Phase 3 — Extension Guide and Code

**Tool:** Claude (claude.ai)

**Prompt used:**
"Give me a final phase3_guide.md using [PDF approach merged with UMAP/HDBSCAN
future work observation] in similar format as the uploaded markdown file,
distributing work between 5 members."

**How it was used:**
- Generated complete Phase 3 guide structure and all five member scripts
- All scripts verified by running locally before submission
- Medical imaging gap justification independently cross-checked against
  Zech et al. 2018 and Oakden-Rayner et al. 2020

**Modifications made:**
- [x] Adjusted N_CLUSTERS and perplexity for actual dataset size
- [x] Filled in XX placeholders in EXTENSION_RESULTS.md with real numbers
- [x] Verified heatmap shapes match expected (200, 3, 224, 224)
- [x] Confirmed border relevance ratio interpretation against cluster_summary.txt