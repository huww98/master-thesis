all: build/main.pdf build/blind.pdf build/sys_design.pdf

.PHONY: all clean draw_figures

build:
	mkdir -p build

build/%.pdf: | build
	latexmk \
		-interaction=nonstopmode \
		-file-line-error \
		-deps-out=$@.deps \
		-outdir=build \
		-xelatex \
		$<

build/main.pdf: main.tex | draw_figures

build/blind.pdf: blind.tex main.tex | draw_figures

build/sys_design.pdf: sys_design.tex

-include build/*.pdf.deps

draw_figures: build/figures/*.pgf

build/figures/*.pgf: draw.py
	python3 $<

clean:
	rm -rf build
