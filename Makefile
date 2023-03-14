all: build/main.pdf build/blind.pdf

-include build/*.pdf.deps

FIGURES = build/figures/problem.pgf \
		  build/figures/one_dim_loss.pgf \
		  build/figures/sdf.pgf \
		  build/figures/sdf_grad.pgf \
		  build/figures/HDRI_stats.pgf \
		  build/figures/l2_loss.pgf

build/main.pdf: main.tex $(FIGURES)
	mkdir -p build
	latexmk \
		-use-make \
		-interaction=nonstopmode \
		-file-line-error \
		-deps-out=build/main.pdf.deps \
		-outdir=build \
		-xelatex \
		$<

build/blind.pdf: blind.tex main.tex $(FIGURES)
	mkdir -p build
	latexmk \
		-use-make \
		-interaction=nonstopmode \
		-file-line-error \
		-deps-out=build/blind.pdf.deps \
		-outdir=build \
		-xelatex \
		$<

build/figures/%.pgf: draw.py
	python3 $<

clean:
	rm -rf build
