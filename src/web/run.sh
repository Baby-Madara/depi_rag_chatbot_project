#!/bin/bash

# 1. do: chmod +x run.sh; ./run.sh

APP_NAME="chatbot"

APP_DIR="$HOME/depi_rag_cahtbot_project/src/web"

PORT=8000



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
echo "Starting Ollama service..."
echo "========================================="

sudo systemctl enable ollama
sudo systemctl restart ollama

echo "========================================="
echo "Pulling model (if not exists)..."
echo "========================================="

ollama list | grep -q "qwen2.5:3b" || ollama pull qwen2.5:3b

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
echo "http://$MY_IP:$PORT"
echo ""

echo "========================================="

echo "To Attach to the session: screen -r chatbot"
echo "Detach from the session (while inside): Press Ctrl+A, then D"
echo "Stop the server: screen -S chatbot -X quit"
echo "Check if running: screen -ls | grep chatbot"
echo "Your server should now be accessible at: http://$MY_IP:80"
echo "========================================="

echo "now if you want a temporarily run: cloudflared tunnel --url http://localhost:80"

echo "========================================="
