# Use a slim Node.js 20 image as base
FROM node:20-slim

# Install Python 3, pip, and system dependencies for OpenCV and AI models
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy the entire workspace to have both backend and ai_service
COPY . .

# --- Backend Setup ---
WORKDIR /app/backend
# Install dependencies
RUN npm install
# Build TypeScript
RUN npm run build

# --- AI Service Setup ---
WORKDIR /app/ai_service
# Create and use a virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip3 install --no-cache-dir -r requirements.txt

# Set environment variables
ENV NODE_ENV=production
ENV PORT=3001
ENV PYTHON_PATH=/opt/venv/bin/python3

# Expose port
EXPOSE 3001

# Return to backend to start the node server
WORKDIR /app/backend

# Final check: build often puts things in dist
CMD ["node", "dist/index.js"]
