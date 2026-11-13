# Jupyter server configuration of the tutorial container. The Nsight GUIs run in the nsys and
# ncu containers; the entrypoint forwards their ports to localhost, where jupyter-server-proxy
# serves them as launcher tiles at /nsys/ and /ncu/.
c = get_config()  # noqa: F821

c.ServerApp.ip = "0.0.0.0"
c.ServerApp.open_browser = False
c.ServerApp.allow_root = True
c.ServerApp.password = ""
c.IdentityProvider.token = ""
c.ServerApp.root_dir = "/"
c.ServerApp.terminado_settings = {"shell_command": ["/bin/bash"]}

c.ServerProxy.servers = {
    "nsys": {
        "command": [],
        "port": 8080,
        "new_browser_tab": False,
        "launcher_entry": {
            "title": "Nsight Systems",
            "category": "Console",  # Console or Notebook: the categories that render icons.
            "icon_path": "/sc2026-tutorial/brev/icons/nsight_systems.svg",
        },
    },
    "ncu": {
        "command": [],
        "port": 8081,
        "new_browser_tab": False,
        "launcher_entry": {
            "title": "Nsight Compute",
            "category": "Console",
            "icon_path": "/sc2026-tutorial/brev/icons/nsight_compute.svg",
        },
    },
}
