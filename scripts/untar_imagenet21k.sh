#!/bin/sh
SRC='./data/winter21_whole'
DEST='./data/imagenet21k'
for tarball in $SRC/*.tar;do
    tardir=$(echo $tarball| cut -d'/' -f 4| cut -d'.' -f 1)
    echo "untarring $tarball to $tardir"
    if [ ! -d $tardir ]; then
        echo "making dir $DEST/$tardir"
        mkdir -p $DEST/$tardir
        tar -xf $tarball -C $DEST/$tardir
    fi
done