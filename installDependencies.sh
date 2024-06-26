#!/usr/bin/env bash

PYTHON_ENV=".venv/bin/activate"
SPINNAKER_DIR="spinnaker_files"



source "$PYTHON_ENV"

pip install --upgrade "numpy<2" matplotlib "Pillow==9.2.0" opencv-python EasyPySpin

mkdir "$SPINNAKER_DIR"
echo "wget -O \"$SPINNAKER_DIR/spinnaker.tar.gz\" \"$SPINNAKER_URL\""
wget -O "$SPINNAKER_DIR/spinnaker.tar.gz" "$SPINNAKER_URL"
echo "tar -xf \"$SPINNAKER_DIR/spinnaker.tar.gz\" -C \"$SPINNAKER_DIR/\""
tar -xf "$SPINNAKER_DIR/spinnaker.tar.gz" -C "$SPINNAKER_DIR/"
echo "pip install $SPINNAKER_DIR/spinnaker_python-*.whl"

pip install $SPINNAKER_DIR/spinnaker_python-4*.whl

# PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Yolov6
git clone https://github.com/meituan/YOLOv6
cd YOLOv6
pip install -r requirements.txt
cd ..


#cleanup
rm -r -f YOLOv6
rm -r "$SPINNAKER_DIR"
