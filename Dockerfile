# Use official Astral UV base image with Python 3.13 on Debian Slim
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# Set working directory inside container
WORKDIR /app

# Enable bytecode compilation and optimization for uv
ENV UV_COMPILE_BYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy dependency definition files first (leverages Docker layer caching)
COPY pyproject.toml uv.lock ./

# Install project dependencies without project source code initially
RUN uv sync --frozen --no-install-project --no-dev

# Copy all project source code into container
COPY . .

# Install the project package itself
RUN uv sync --frozen --no-dev

# Expose default port (Render will override PORT at runtime)
EXPOSE 8000

# Run FastAPI server
CMD ["uv", "run", "python", "run_server.py"]
