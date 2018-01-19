# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = Serendipyty
SOURCEDIR     = source
# BUILDDIR      = ../../serendipyty.github.io
BUILDDIR      = build
PDFBUILDDIR   = /tmp
PDF			  =	../manual.pdf
WEBSITEDIR      = ../../serendipyty.github.io

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

commit:
	cp -a build/html/. "$(WEBSITEDIR)"; cd "$(WEBSITEDIR)"; git add .; git commit -m "Rebuilt docs"; git push
