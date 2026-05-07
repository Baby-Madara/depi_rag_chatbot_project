#!/bin/bash

# 1. do: chmod +x setup.sh; ./setup.sh
# 2. Before running: upload project into:
# /home/azureuser/depi_rag_cahtbot_project/src/web
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

sudo apt install -y
python3.11
python3.11-dev
python3.11-distutils
python3-pip
nginx
git

echo "========================================="
echo "Python version:"
echo "========================================="

python3.11 --version

echo "========================================="
echo "Installing Python packages globally..."
echo "========================================="

pip3 install --break-system-packages
flask
gunicorn
python-dotenv

if [ -f "$APP_DIR/requirements.txt" ]; then

```
pip3 install --break-system-packages \
-r "$APP_DIR/requirements.txt"
```

else

```
echo "WARNING: requirements.txt not found"
```

fi

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

```
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
```

}
EOF

sudo ln -sf
/etc/nginx/sites-available/$APP_NAME
/etc/nginx/sites-enabled/$APP_NAME

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
echo "Starting chatbot service..."
echo "========================================="

sudo systemctl daemon-reload

sudo systemctl enable $APP_NAME

sudo systemctl restart $APP_NAME

echo "========================================="
echo "DONE!"
echo "========================================="

echo ""

echo "Useful commands:"

echo "-----------------------------------------"

echo "Check app status:"
echo "sudo systemctl status $APP_NAME"

echo ""

echo "Restart app:"
echo "sudo systemctl restart $APP_NAME"

echo ""

echo "Live logs:"
echo "journalctl -u $APP_NAME -f"

echo ""

echo "Restart nginx:"
echo "sudo systemctl restart nginx"

echo ""

echo "Your server should now be accessible at:"
echo "http://YOUR_VM_IP"

echo ""
