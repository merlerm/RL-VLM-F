export PYTHONPATH=${PWD}:PYTHONPATH
export PYTHONPATH=${PWD}/softgym:$PYTHONPATH
export PYFLEXROOT=${PWD}/softgym/PyFlex
export PYTHONPATH=${PYFLEXROOT}/bindings/build:$PYTHONPATH
export LD_LIBRARY_PATH=${PYFLEXROOT}/external/SDL2-2.0.4/lib/x64:$LD_LIBRARY_PATH
export MUJOCO_GL=egl
eval "$(micromamba shell hook --shell bash)"
micromamba activate rlvlmf
