FROM nginx:alpine

# Copy our custom Nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy the web project files to the Nginx html directory
COPY web/ /usr/share/nginx/html/

EXPOSE 108