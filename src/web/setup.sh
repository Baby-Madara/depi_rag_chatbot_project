#!/bin/bash

# 1. do: chmod +x setup.sh; ./setup.sh
# 2. Before running: upload project into:
# /home/noob/depi_rag_cahtbot_project/src/web
# 3. make sure: {server.py, requirements.txt, setup.sh} exist there.
set -e

APP_NAME="chatbot"

APP_DIR="$HOME/depi_rag_cahtbot_project/src/web"

PORT=8000

echo "========================================="
echo "Updating Ubuntu..."
echo "========================================="

sudo apt update
sudo apt upgrade -y

echo "========================================="
echo "Installing Python 3.11 + dependencies..."
echo "========================================="

sudo apt install -y software-properties-common

sudo add-apt-repository ppa:deadsnakes/ppa -y

sudo apt update

sudo apt install -y \
python3.11 \
python3.11-dev \
python3.11-distutils \
python3-pip \
nginx \
git \
screen \
zstd \
langchain-openai \
langchain_community 

echo "========================================="
echo "Python version:"
echo "========================================="

python3.11 --version

echo "========================================="
echo "Installing Python packages globally..."
echo "========================================="

pip3 install --break-system-packages \
flask \
gunicorn \
python-dotenv 

if [ -f "$APP_DIR/requirements.txt" ]; then
pip3 install --break-system-packages -r "$APP_DIR/requirements.txt"
else
echo "WARNING: requirements.txt not found"
fi


echo "========================================="
echo "Checking Ollama installation..."
echo "========================================="

if ! command -v ollama &> /dev/null; then
echo "Ollama not found. Installing..."
curl -fsSL https://ollama.com/install.sh | sh
else
echo "Ollama already installed. Skipping..."
fi

echo "========================================="
echo "Starting Ollama service..."
echo "========================================="

sudo systemctl enable ollama
sudo systemctl restart ollama

echo "========================================="
echo "Pulling model (if not exists)..."
echo "========================================="

ollama list | grep -q "qwen2.5:3b" || ollama pull qwen2.5:3b

echo "========================================="
echo "Restarting chatbot service..."
echo "========================================="

sudo systemctl restart $APP_NAME

echo "========================================="
echo "DONE!"
echo "========================================="

echo "========================================="
echo "Creating systemd service..."
echo "========================================="

sudo tee /etc/systemd/system/$APP_NAME.service > /dev/null <<EOF
[Unit]
Description=Flask Chatbot App
After=network.target

[Service]
User=$USER
WorkingDirectory=$APP_DIR

ExecStart=/usr/bin/python3.11
-m gunicorn
-w 4
-b 127.0.0.1:$PORT
server:app

Restart=always

[Install]
WantedBy=multi-user.target
EOF

echo "========================================="
echo "Configuring Nginx..."
echo "========================================="

sudo tee /etc/nginx/sites-available/$APP_NAME > /dev/null <<EOF
server {
listen 80 default_server;
server_name _;

client_max_body_size 50M;

location / {
    proxy_pass http://127.0.0.1:$PORT;

    proxy_set_header Host \$host;
    proxy_set_header X-Real-IP \$remote_addr;
    proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;

    proxy_set_header Upgrade \$http_upgrade;
    proxy_set_header Connection "upgrade";
}

}
EOF

sudo ln -sf /etc/nginx/sites-available/$APP_NAME /etc/nginx/sites-enabled/$APP_NAME 

sudo rm -f /etc/nginx/sites-enabled/default

echo "========================================="
echo "Testing Nginx config..."
echo "========================================="

sudo nginx -t

echo "========================================="
echo "Restarting Nginx..."
echo "========================================="

sudo systemctl restart nginx

echo "========================================="
echo "Starting chatbot server in background..."
echo "========================================="

screen -dmS chatbot python3 server.py

echo "========================================="
echo "DONE!"
echo "========================================="

echo ""

echo "Useful commands:"

echo "-----------------------------------------"

echo "Attach to server session:"
echo "screen -r chatbot"

echo ""

echo "Detach from session (inside screen):"
echo "Ctrl+A, D"

echo ""

echo "Stop server:"
echo "screen -S chatbot -X quit"

echo ""

echo "Check if running:"
echo "screen -ls | grep chatbot"

echo ""

MY_IP=$(curl -s ifconfig.me)
echo "Your server should now be accessible at:"
echo "http://$MY_IP:80"
echo ""

echo "========================================="

echo "To Attach to the session: screen -r chatbot"
echo "Detach from the session (while inside): Press Ctrl+A, then D"
echo "Stop the server: screen -S chatbot -X quit"
echo "Check if running: screen -ls | grep chatbot"
echo "Your server should now be accessible at: http://$MY_IP:80"
echo "========================================="
