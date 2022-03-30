FROM wandb/local:latest

USER root
ADD chown_skript.sh .
RUN chmod +x chown_skript.sh
RUN ./chown_skript.sh etc
RUN ./chown_skript.sh var
RUN ./chown_skript.sh vol/env
RUN touch /run/nginx.pid && chown -R 1147:1003 /run/nginx.pid
RUN chown -R 1147:1003 .
USER wandb