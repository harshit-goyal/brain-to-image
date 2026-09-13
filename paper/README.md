# arXiv submission package

The manuscript is an eleven-page research paper reporting the measured
THINGS-EEG2 retrieval experiment, three-seed and robustness analyses, and
negative results from direct generation. It deliberately does not describe
pretrained diffusion output as direct brain-image reconstruction.

## Build

```bash
mkdir -p build
pdflatex -output-directory=build main.tex
BIBINPUTS=.: bibtex build/main
pdflatex -output-directory=build main.tex
pdflatex -output-directory=build main.tex
```

The compiled paper is `brain-to-image-paper.pdf`. The upload-ready source archive is
`arxiv-submission.zip`.

## arXiv metadata

- **Primary category:** `cs.CV`
- **Cross-list:** `q-bio.NC`
- **Author:** Harshit Goyal
- **Affiliation:** BITS Pilani, India
- **Comments:** 11 pages, 6 figures, 7 tables. Code and reproducibility artifacts available
  at <https://github.com/harshit-goyal/brain-to-image>.
- **Suggested license:** CC BY 4.0, matching the author's prior arXiv submission

Before final submission, verify the compiled PDF, author metadata, category selection,
license, and all claims in the arXiv preview. The author must complete submission through
their own arXiv account.
