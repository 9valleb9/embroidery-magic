# Embroidery Magic with Ink/Stitch built in, so machine-file export works out of the box.
#   docker build -t embroidery-magic .
#   docker run --rm -p 8765:8765 embroidery-magic      → http://localhost:8765
FROM python:3.11-slim

ARG INKSTITCH_VERSION=3.3.0
ARG TARGETARCH

# cairo for SVG input; the GTK/X libs are what Ink/Stitch's bundled runtime links against,
# and Xvfb gives it the (virtual) display it insists on even for command-line export
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates xz-utils libcairo2 libglib2.0-0 libgl1 libgtk-3-0 libsm6 libxext6 \
        xvfb xauth \
    && rm -rf /var/lib/apt/lists/*

RUN case "$TARGETARCH" in arm64) arch=aarch64 ;; *) arch=x86_64 ;; esac \
    && curl -fsSL "https://github.com/inkstitch/inkstitch/releases/download/v${INKSTITCH_VERSION}/inkstitch-${INKSTITCH_VERSION}-linux-${arch}.tar.xz" \
       | tar -xJ -C /opt \
    && printf '#!/bin/sh\nexec xvfb-run -a /opt/inkstitch/bin/inkstitch "$@"\n' > /usr/local/bin/inkstitch \
    && chmod +x /usr/local/bin/inkstitch

ENV INKSTITCH_BIN=/usr/local/bin/inkstitch \
    HOST=0.0.0.0 PORT=8765 NO_BROWSER=1 PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 8765
CMD ["python", "app.py"]
