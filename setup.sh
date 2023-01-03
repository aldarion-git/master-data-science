mkdir -p ~/.streamlit/
echo "[server]"  > ~/.streamlit/config.toml 
echo "headless = true"  >> ~/.streamlit/config.toml
echo "port = $PORT"  >> ~/.streamlit/config.toml
echo "enableCORS = false"  >> ~/.streamlit/config.toml
echo "[theme]"  > ~/.streamlit/config.toml
echo "primaryColor='#0B0078'"  >> ~/.streamlit/config.toml
echo "backgroundColor='#FFF'"  >> ~/.streamlit/config.toml
echo "secondaryBackgroundColor='#F0F2F6'"  >> ~/.streamlit/config.toml
echo "textColor='#06003F'"  >> ~/.streamlit/config.toml
echo "font='sans serif'"  >> ~/.streamlit/config.toml
echo "base='light'"  >> ~/.streamlit/config.toml