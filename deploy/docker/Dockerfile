FROM ubuntu:24.04

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    sudo \
    curl \
    git \
    jq \
    tar \
    unzip \
    ca-certificates \
    build-essential \
    golang \
    procps \
    python3 \
    python3-pip \
    python3-venv \
    vim \
    tmux \
    zsh \
    apt-transport-https \
    gnupg \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

# Install Neovim 0.11+ from GitHub releases using curl (wget not available yet at this stage).
# Must be done before setup.sh runs so setup.sh's version check skips the neovim install step.
# setup.sh uses wget for its own neovim download, which isn't installed until later in that script,
# causing it to silently fall back to the PPA which provides an older version.
RUN curl -fsSL https://github.com/neovim/neovim/releases/download/stable/nvim-linux-x86_64.tar.gz \
    -o /tmp/nvim.tar.gz && \
    tar xzf /tmp/nvim.tar.gz -C /opt/ && \
    ln -sf /opt/nvim-linux-x86_64/bin/nvim /usr/local/bin/nvim && \
    rm /tmp/nvim.tar.gz

# Install Google Cloud CLI
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    apt-get update && \
    apt-get install -y google-cloud-cli && \
    rm -rf /var/lib/apt/lists/*


# Setup the coder user account
ARG USER=coder
RUN useradd --groups sudo --shell /bin/zsh ${USER} \
    && echo "${USER} ALL=(ALL) NOPASSWD:ALL" >/etc/sudoers.d/${USER} \
    && chmod 0440 /etc/sudoers.d/${USER}

# Setup user account and home directory in the image
RUN mkdir -p /home/${USER}/.config/gcloud && \
    chown -R ${USER}:${USER} /home/${USER}

# Run dotfiles setup at build time so workspace startup is just a fast file copy.
# We run setup.sh as root with HOME pointed at /home/coder so oh-my-zsh, p10k,
# plugins, and config files all land in the right place.
# Patch p10k to ascii mode in the source BEFORE setup.sh copies it — setup.sh uses
# set -euo pipefail and may exit early, so patching the destination after the fact
# is unreliable (the file may not exist yet, and the || true hides the failure).
RUN git clone https://github.com/amrit110/dotfiles /tmp/dotfiles && \
    git -C /tmp/dotfiles checkout coder && \
    sed -i 's/POWERLEVEL9K_MODE=.*/POWERLEVEL9K_MODE=ascii/' /tmp/dotfiles/.p10k.zsh && \
    cd /tmp/dotfiles && \
    HOME=/home/${USER} DEBIAN_FRONTEND=noninteractive bash setup.sh || \
    echo "Warning: setup.sh had non-fatal errors, continuing build"

# Safety-net: if setup.sh exited before copy_dotfiles, copy it explicitly now.
RUN cp -n /tmp/dotfiles/.p10k.zsh /home/${USER}/.p10k.zsh 2>/dev/null || true

# Fix ownership, then snapshot to /opt/home-seed for runtime seeding
RUN chown -R ${USER}:${USER} /home/${USER} && \
    mkdir -p /opt/home-seed && \
    cp -a /home/${USER}/. /opt/home-seed/

# Note: We don't set USER here because the container needs to run as root
# to fix permissions on the mounted volume. The container command will
# chown /home/coder and then su to the coder user.
