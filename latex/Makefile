.PHONY: all clean results plots figures

LATEXMK  = latexmk

all: main.pdf

main.pdf: main.tex results_table.tex figures/summary_barchart.pdf
	$(LATEXMK) -pdf -interaction=nonstopmode main.tex

results: ../results/summary.json
	cd .. && python latex/generate_latex.py \
		--results-dir results \
		--output latex/results_table.tex

plots:
	cd .. && python latex/plot_results.py results latex/figures

figures: plots

clean:
	rm -f *.aux *.log *.out *.toc *.bbl *.blg *.fdb_latexmk *.fls
	rm -f main.pdf results_table.tex
	rm -rf figures/
	rm -f plot_results.py
