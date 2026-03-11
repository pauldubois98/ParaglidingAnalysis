FROM nginx:alpine

# Create the cache directory and set permissions
RUN mkdir -p /var/cache/nginx && chown -R nginx:nginx /var/cache/nginx

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY web/ /usr/share/nginx/html/

EXPOSE 108