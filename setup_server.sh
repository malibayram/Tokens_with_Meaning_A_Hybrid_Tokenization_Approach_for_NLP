#!/bin/bash
set -e

# scp requirements.txt verda-h100:~/
# scp setup_server.sh verda-h100:~/
# scp .env verda-h100:~/
# scp embedding_trainer.py verda-h100:~/
# scp train.py verda-h100:~/

echo "1. Installing python3-venv and pip..."
apt update
apt install -y python3-venv python3-pip

echo "2. Creating virtual environment..."
python3 -m venv venv

echo "3. Activating venv and installing dependencies..."
source venv/bin/activate
pip install --upgrade pip

# Install other requirements
echo "   Installing other requirements..."
pip install -r requirements.txt

echo "Done! Run 'source venv/bin/activate' to start."
echo "Then to run the training script, use: 'nohup python train.py > train.log 2>&1 &'"
