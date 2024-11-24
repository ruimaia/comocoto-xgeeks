FROM ghcr.io/astral-sh/uv:python3.12-bookworm
WORKDIR /app
COPY . .
RUN uv sync
ENTRYPOINT ["uv", "run", "python", "src/comocoto_xgeeks/app.py"]