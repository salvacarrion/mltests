# Resizes images to X, whose sizes is greater than X
SIZE=512; magick mogrify -resize "$SIZE>x$SIZE>" -format png -verbose -path ../images$SIZE *.png