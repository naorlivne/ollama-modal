import modal
import os
import subprocess
import time

from modal import build, enter, method

MODEL = os.environ.get("MODEL", "llama3.2")

def pull(model: str = MODEL):
    subprocess.run(["systemctl", "daemon-reload"])
    subprocess.run(["systemctl", "enable", "ollama"])
    subprocess.run(["systemctl", "start", "ollama"])
    time.sleep(5)  # 2s, wait for the service to start
    subprocess.run(["ollama", "pull", model], stdout=subprocess.PIPE)

image = (
    modal.Image
    .debian_slim()
    .apt_install("curl", "systemctl")
    .run_commands( # from https://github.com/ollama/ollama/blob/main/docs/linux.md
        "curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz",
        "tar -C /usr -xzf ollama-linux-amd64.tgz",
        "chmod +x /usr/bin/ollama",
        "useradd -r -s /bin/false -m -d /usr/share/ollama ollama",
    )
    .copy_local_file("ollama.service", "/etc/systemd/system/ollama.service")
    .pip_install("ollama")
    .run_function(pull)
)

app = modal.App(name="ollama", image=image)

with image.imports():
    import ollama

@app.cls(gpu="a10g", container_idle_timeout=300)
class Ollama:

    @enter()
    def load(self):
        subprocess.run(["systemctl", "start", "ollama"])

    @method()
    def infer(self, text: str):
        stream = ollama.chat(
            model=MODEL,
            messages=[{'role': 'user', 'content': text}],
            stream=True
        )
        for chunk in stream:
            yield chunk['message']['content']
            print(chunk['message']['content'], end='', flush=True)
        return

# Convenience thing, to run using:
#
#  $ modal run ollama-modal.py [--lookup] [--text "Why is the sky blue?"]
@app.local_entrypoint()
def main(text: str = "Why is the sky blue?", lookup: bool = False):
    if lookup:
        ollama = modal.Cls.lookup("ollama", "Ollama")
    else:
        ollama = Ollama()
    for chunk in ollama.infer.remote_gen(text):
        print(chunk, end='', flush=False)
