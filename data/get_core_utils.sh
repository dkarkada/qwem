#!/bin/bash

target_dir="$HOME/pythonlib/expt-core"
if [ -d "$target_dir" ]; then
    (
    cd "$target_dir" && git pull
    )
else
    git clone "https://github.com/dkarkada/expt-core" "${target_dir}"
fi
export PYTHONPATH="$PYTHONPATH:${target_dir}/"
export DATASETPATH="$HOME/datasets"
export EXPTPATH="$HOME/experiments"

# >>> uncomment and copy-paste the following to your .bashrc
# export DATASETPATH="$HOME/datasets"
# export EXPTPATH="$HOME/experiments"
# target_dir="$HOME/pythonlib/expt-core"
# if [ -d "$target_dir" ]; then
#     export PYTHONPATH="$PYTHONPATH:${target_dir}/"
# else
#     echo "Could not find repo expt-core."
# fi
# unset target_dir