all: build/main.pdf

-include build/main.pdf.deps

build/main.pdf: main.tex build/figures/problem.pgf build/figures/one_dim_loss.pgf
	mkdir -p build
	latexmk \
		-use-make \
		-interaction=nonstopmode \
		-file-line-error \
		-deps-out=build/main.pdf.deps \
		-outdir=build \
		-xelatex \
		$<

build/figures/%.pgf: draw.py
	python3 $<

clean:
	rm -rf build
