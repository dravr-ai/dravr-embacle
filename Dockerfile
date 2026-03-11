# ABOUTME: Multi-stage Docker build for embacle-server and embacle-mcp binaries
# ABOUTME: Runtime includes Node.js for installing npm-based CLI backends

FROM rust:1-bookworm AS builder
WORKDIR /build
COPY . .
RUN cargo build --release -p embacle-server -p embacle-mcp

FROM node:22-bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash embacle

COPY --from=builder /build/target/release/embacle-server /usr/local/bin/
COPY --from=builder /build/target/release/embacle-mcp /usr/local/bin/

USER embacle
WORKDIR /home/embacle

EXPOSE 3000
ENTRYPOINT ["embacle-server"]
CMD ["--host", "0.0.0.0"]
