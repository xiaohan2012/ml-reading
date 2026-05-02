cd figures/prior_samples;
for i in *.svg; do
  inkscape $i --export-area-drawing --batch-process --export-type=pdf --export-filename=${i%.svg}_svg-tex.pdf
done
sed -i -E "s/includesvg/includeinkscape/" main.tex;
sed -i -E "s/\.svg/_svg-tex.pdf/" main.tex;