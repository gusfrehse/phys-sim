#!/bin/bash

FRAGMENTS=basic_frag.glsl
VERTEX=basic_vert.glsl

GLSLC_FLAGS=-g

OUT_HEADER=shaders.h

rm -f $OUT_HEADER
echo -e '#ifndef SHADERS_H\n#define SHADERS_H' > $OUT_HEADER

for s in $FRAGMENTS ; do
    out=$(basename $s .glsl).spv
    glslc $GLSLC_FLAGS -fshader-stage=fragment $s -o $out
    xxd -i $out >> $OUT_HEADER ; \
done

for s in $VERTEX ; do
    out=$(basename $s .glsl).spv
    glslc $GLSLC_FLAGS -fshader-stage=vertex $s -o $out
    xxd -i $out >> $OUT_HEADER ; \
done

echo -e '#endif\n' >> $OUT_HEADER
