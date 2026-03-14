# Paragliding Heatmap

## Development
Local server via
```bash
python server.py
```
Then http://localhost:8000 (port forward may be needed).

Formatting all HTML files:
```bash
$ npx js-beautify --type html --indent-size 2 --replace web/*.html
```

## Production
Build via
```bash
$ docker compose down
$ docker compose build
```
Serve via
```bash
$ docker compose up
```
Then http://localhost:108 (port forward may be needed).
Or https://paragliding.duckdns.org/ (if main NGINX is not broken).

## Sites
- La Seranne: https://paragliding.duckdns.org/area-selection?bbox=43.839100,43.901727,3.601400,3.713722&to=43.869100,3.640600&la=43.895500,3.699840
- Arbas: https://paragliding.duckdns.org/area-selection?bbox=42.961060,43.014963,0.845205,0.937002&to=42.969398,0.885938&la=42.992160,0.903981
