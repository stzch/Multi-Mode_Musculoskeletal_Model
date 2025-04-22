#!/usr/bin/env bash
ENVDIR=${ENVDIR:-~/pkgenv }

# If CentOS server, set C++ compiler manually
if [ -f /etc/redhat-release ]; then
    export CC=/opt/ohpc/pub/compiler/gcc/8.3.0/bin/gcc
    export CXX=/opt/ohpc/pub/compiler/gcc/8.3.0/bin/g++
fi

mkdir -p build
pushd build

PYTHON_EXECUTABLE=/home/chen/miniconda3/pkgs/python-3.6.13-h12debd9_1/bin/python3.6
PYTHON_INCLUDE_DIR=/home/chen/miniconda3/pkgs/python-3.6.13-h12debd9_1/include/python3.6m
PYTHON_LIBRARY=/home/chen/miniconda3/pkgs/python-3.6.13-h12debd9_1/lib/libpython3.6m.so

cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH=$ENVDIR \
      -DCMAKE_INSTALL_PREFIX=$ENVDIR \
      -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=TRUE \
      -DCMAKE_INSTALL_RPATH=$ENVDIR \
      -DFCL_INCLUDE_DIRS=$ENVDIR/include/fcl \
      -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE} \
      -DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIR} \
      -DPYTHON_LIBRARY=${PYTHON_LIBRARY} \
      ..
popd
