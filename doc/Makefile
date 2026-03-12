# Simple Makefile for building the pySimNMR manual with latexmk

TEX=pySimNMR_manual

.PHONY: all pdf clean distclean watch

all: pdf

pdf:
	latexmk -pdf $(TEX).tex

watch:
	latexmk -pdf -pvc $(TEX).tex

clean:
	latexmk -c $(TEX).tex

distclean:
	latexmk -C $(TEX).tex
