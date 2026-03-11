# Paragliding Heatmap

## Development
Local server via
```bash
python server.py
```
Then http://localhost:8000 (port forward may be needed).

## Production
Build via
```bash
$ docker compose up --build
```
Serve via
```bash
$ docker compose up
```
Then http://localhost:108 (port forward may be needed).
Or https://paragliding.duckdns.org/ (if main NGINX is not broken).
