# IB031 - Zelovoc

## Navigation

Models are in respective notebooks, the data analysis is in `data_analysis.ipynb`
## Generating slides from the notebook

To generate slides from `presentation.ipynb` (e.g. after editting the notebook):

1. `pip install nbconvert==5.6.1`
2. `jupyter nbconvert presentation.ipynb --to slides --TemplateExporter.exclude_input=True`
3. output should be in `presentation.slides.html`


