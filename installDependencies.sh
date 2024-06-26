#!/usr/bin/env bash
#Repo setup for x86_64 Ubuntu Linex

PYTHON_ENV_NAME=".venv"
PYTHON_ENV="$PYTHON_ENV_NAME/bin/activate"
SPINNAKER_DIR="spinnaker_files"
PYTHON_COMMAND="python3"
PIP_COMMAND="$PYTHON_COMMAND -m pip"


global_install=''

# install flags
env='true'
pip='true'
spinnaker='true'
torch='true'
yolo='true'
yolo_version=6
cleanup='true'

# pip flags
$PIP_FLAGS='--upgrade'

# handle args
print_usage() {
  printf "Usage: 
  -e: Disable create environment
  -p: ...
  -s: Disable install Sinnaker SDk
  -t: Disable install torch
  -y: Disable install yolo
  -c: Disable cleanup
  "
}

while getopts 'epstyc:' flag; do
  case "${flag}" in
    e) env='' ;;
    p) pip='' ;;
    s) spinnaker='' ;;
    t) torch='' ;;
    y) yolo='' ;;
    c) cleanup='' ;;
    *) print_usage
       exit 1 ;;
  esac
done



if [[ -n "$global_install" ]]; then
    env=''
fi



if [[ -n "$env" ]]; then
    if [ ! -f ./PYTHON_ENV_NAME ]; then
        echo "File $PYTHON_ENV_NAME not found! Creating..."
        $PYTHON_COMMAND -m venv $PYTHON_ENV_NAME
    fi
fi


if [[ -z "$global_install" ]]; then
    source "$PYTHON_ENV"
fi


if [[ -n "$spinnaker" ]]; then
    $PIP_COMMAND install "numpy<2" matplotlib "Pillow==9.2.0" opencv-python EasyPySpin

    if [[ -n "$SPINNAKER_URL" ]]; then
        mkdir "$SPINNAKER_DIR"
        echo "wget -O \"$SPINNAKER_DIR/spinnaker.tar.gz\" \"$SPINNAKER_URL\""
        wget -O "$SPINNAKER_DIR/spinnaker.tar.gz" "$SPINNAKER_URL"
        echo "tar -xf \"$SPINNAKER_DIR/spinnaker.tar.gz\" -C \"$SPINNAKER_DIR/\""
        tar -xf "$SPINNAKER_DIR/spinnaker.tar.gz" -C "$SPINNAKER_DIR/"
        echo "pip install $SPINNAKER_DIR/spinnaker_python-*.whl"

        $PIP_COMMAND install $PIP_FLAGS $SPINNAKER_DIR/spinnaker_python-4*.whl
    else
        echo "Missing SPINNAKER_URL env variable."
    fi
fi


if [[ -n "$torch" ]]; then
    # PyTorch
    $PIP_COMMAND install $PIP_FLAGS torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
fi


if [[ -n "$yolo" ]]; then
    if (( $yolo_version == 6 )); then
        # Yolov6
        git clone https://github.com/meituan/YOLOv6
        cd YOLOv6
        $PIP_COMMAND install $PIP_FLAGS -r requirements.txt
        cd ..
    fi
fi


if [[ -n "$cleanup" ]]; then
    #cleanup
    rm -r -f YOLOv6
    rm -r "$SPINNAKER_DIR"
fi