# Use Python with Rust support
FROM python:3.11-slim as builder

# Install Rust and build tools
RUN apt-get update && apt-get install -y \
    curl build-essential pkg-config libssl-dev \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app
COPY . .

# Install maturin and build the rust extension
# We don't need to install dependencies here if we just want the wheel
RUN pip install --no-cache-dir maturin
RUN maturin build --release

# Final image
FROM python:3.11-slim
WORKDIR /app

# Copy the built wheel from builder and install it
COPY --from=builder /app/target/wheels/*.whl .
RUN pip install *.whl numpy gradio sentence-transformers

# Copy the rest of the app
COPY app.py .
COPY README.md .

# Hugging Face Spaces typically use port 7860
EXPOSE 7860
CMD ["python", "app.py"]
