# SBTJ Study 1 supplementary materials

This folder contains the data, analysis code, stimulus-generation code, and stimulus files needed to reproduce [insert paper title].

## Contents

```text
Supplementary_Materials/
|-- README.md
|-- analysis.Rmd
|-- generate_stimuli.py
|-- data/
|   |-- data_pilot_raw.csv
|   `-- Thesis_Pilot_QualtricsDataFinal.csv
`-- stimuli/
    |-- Stimuli_Info.csv
    |-- bars_pos/
    |-- bars_neg/
    |-- scatter_pos/
    `-- scatter_neg/
```

## Reproduce the analyses

Use R 4.6.1 or a compatible version with these packages:

- dplyr
- readr
- tibble
- lme4
- lmerTest
- emmeans
- ggplot2
- rmarkdown
- knitr
- scales

From the `Supplementary_Materials` folder, run:

```r
rmarkdown::render("analysis.Rmd")
```

The rendered HTML file is written beside the R Markdown file. Machine-readable result tables and figures are written under `outputs/`.

## Reproduce the stimuli

Use Python 3 with Pillow, NumPy, and pandas. From the `Supplementary_Materials` folder, run:

```bash
python generate_stimuli.py --outdir reproduced_stimuli --seed 42
```

`stimuli/Stimuli_Info.csv` contains one row for each of the stimulus images. 
