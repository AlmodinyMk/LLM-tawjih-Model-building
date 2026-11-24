# The image ClusterFuzzLite builds the fuzzers in (see build.sh).
FROM gcr.io/oss-fuzz-base/base-builder-python
COPY . $SRC/tawjih
WORKDIR $SRC/tawjih
COPY .clusterfuzzlite/build.sh $SRC/
