"""gix: GIX GPU cluster from your laptop (macOS, Linux, Windows).

Only `gix setup` runs locally. Every other command is forwarded to `gix` on the
login host over SSH, so the cluster side can change without students updating.
"""
import argparse
import getpass
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from . import __version__

LOGIN_ADDR = os.environ.get("GIX_LOGIN_ADDR", "10.155.32.7")
LOGIN = "gix-login"
GPU = "gix-gpu"
REMOTE_GIX = os.environ.get("GIX_REMOTE_CMD", "gix")
IS_WIN = os.name == "nt"

SSH_DIR = Path.home() / ".ssh"
SSH_CONFIG = SSH_DIR / "config"
KEY = SSH_DIR / "id_ed25519"
BEGIN = "# --- GIX GPU cluster (managed by `gix setup`) ---"
END = "# --- end GIX ---"
# blocks written by older setup tools; replaced by ours
OLD_BLOCKS = [
    ("# --- GIX GPU cluster", "# --- end GIX ---"),
    ("# >>> GIX GPU", "# <<< GIX GPU <<<"),
]


def ssh_exe():
    exe = shutil.which("ssh")
    if not exe:
        sys.exit("gix: `ssh` not found. On Windows: Settings > Apps > Optional features > OpenSSH Client.")
    return exe


def quote_path(p):
    p = str(p)
    return f'"{p}"' if " " in p else p


def mux_path(netid):
    # Windows OpenSSH can't multiplex; macOS refuses socket paths over ~104 bytes
    if IS_WIN:
        return None
    p = SSH_DIR / f"gix-cm-{netid}"
    return p if len(str(p).encode()) <= 100 else None


def config_block(netid):
    key = quote_path(KEY)
    proxy_ssh = quote_path(ssh_exe()) if IS_WIN else "ssh"
    lines = [
        BEGIN,
        f"Host {LOGIN}",
        f"    HostName {LOGIN_ADDR}",
        f"    User {netid}",
        f"    IdentityFile {key}",
        "    IdentitiesOnly yes",
        "    StrictHostKeyChecking accept-new",
    ]
    mux = mux_path(netid)
    if mux:
        # one shared login for all connections (each new login costs a NetID lookup)
        lines += ["    ControlMaster auto", f"    ControlPath {quote_path(mux)}", "    ControlPersist 10m"]
    lines += [
        "",
        f"Host {GPU}",
        "    HostName gix-gpu-job",
        f"    User {netid}",
        f"    IdentityFile {key}",
        "    IdentitiesOnly yes",
        f'    ProxyCommand {proxy_ssh} -q {LOGIN} "/usr/local/bin/gix-devjob"',
        "    # you can land on any GPU node, each with its own host key",
        "    StrictHostKeyChecking no",
        f"    UserKnownHostsFile {'NUL' if IS_WIN else '/dev/null'}",
        "    LogLevel ERROR",
        "    ServerAliveInterval 30",
        END,
        "",
    ]
    return "\n".join(lines)


def strip_blocks(text):
    for begin, end in OLD_BLOCKS:
        pattern = re.compile(r"^" + re.escape(begin) + r".*?^" + re.escape(end) + r"[^\n]*\n?", re.S | re.M)
        text = pattern.sub("", text)
    return text.lstrip("\n")


def write_ssh_config(netid):
    SSH_DIR.mkdir(mode=0o700, exist_ok=True)
    old = SSH_CONFIG.read_text() if SSH_CONFIG.exists() else ""
    if old and not (SSH_DIR / "config.gix-backup").exists():
        (SSH_DIR / "config.gix-backup").write_text(old)
    # ssh uses the first matching Host, so our block goes on top
    SSH_CONFIG.write_text(config_block(netid) + "\n" + strip_blocks(old))
    if not IS_WIN:
        SSH_CONFIG.chmod(0o600)


def ssh(args, **kw):
    # -F: always the config `gix setup` wrote (ssh itself ignores $HOME on Linux/macOS)
    return subprocess.run([ssh_exe(), "-F", str(SSH_CONFIG), *args], **kw)


def key_works():
    r = ssh(["-o", "BatchMode=yes", "-o", "ConnectTimeout=15", LOGIN, "true"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return r.returncode == 0


def cmd_setup(args):
    netid = args.netid or input("UW NetID: ").strip()
    if not re.fullmatch(r"[a-z][a-z0-9]{0,15}", netid):
        sys.exit(f"gix: '{netid}' doesn't look like a NetID")

    if not KEY.exists():
        print(f"Creating SSH key {KEY}")
        SSH_DIR.mkdir(mode=0o700, exist_ok=True)
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(KEY)], check=True)

    write_ssh_config(netid)
    print(f"Added {LOGIN} and {GPU} to {SSH_CONFIG}", flush=True)

    if key_works():
        print("Your key is already set up on the cluster.")
    else:
        print("Copying your key to the cluster. Enter your NetID password:", flush=True)
        pub = KEY.with_suffix(".pub").read_text()
        r = ssh([LOGIN, "umask 077; mkdir -p ~/.ssh; cat >> ~/.ssh/authorized_keys"], input=pub, text=True)
        if r.returncode != 0 or not key_works():
            sys.exit("gix: could not log in. Are you on the UW network (or Husky OnNet)? Is the password right?")

    print(flush=True)
    ssh([LOGIN, REMOTE_GIX, "whoami"])
    print("\nDone. Try:  gix status")
    return 0


def nas_path(rel=""):
    """Absolute path on the cluster for a path relative to the student's NAS folder."""
    args = [REMOTE_GIX, "nas-path"] + ([rel] if rel else [])
    r = ssh(["-o", "ConnectTimeout=20", LOGIN, " ".join(shlex.quote(a) for a in args)],
            stdout=subprocess.PIPE, text=True)
    if r.returncode != 0 or not r.stdout.strip():
        sys.exit(r.returncode or 1)
    return r.stdout.strip()


def scp(src, dst):
    exe = shutil.which("scp")
    if not exe:
        sys.exit("gix: `scp` not found (it comes with the OpenSSH client).")
    return subprocess.run([exe, "-F", str(SSH_CONFIG), "-r", src, dst]).returncode


def cmd_upload(a):
    local = Path(a.path)
    if not local.exists():
        sys.exit(f"gix: {local} not found")
    dest = nas_path(a.to or "")
    print(f"Uploading {local} -> {dest}/{local.resolve().name}", flush=True)
    rc = scp(str(local), f"{LOGIN}:{dest}/")
    if rc == 0:
        rel = f"{a.to.rstrip('/')}/" if a.to else ""
        print(f"Done. On the cluster it is: {rel}{local.resolve().name}")
    return rc


def cmd_download(a):
    src = nas_path(a.path)
    local = Path(a.to or ".")
    # scp renames instead of copying into a folder that doesn't exist yet
    local.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {src} -> {local.resolve()}", flush=True)
    return scp(f"{LOGIN}:{src}", str(local))


def forward(argv, tty=False):
    remote = " ".join(shlex.quote(a) for a in [REMOTE_GIX, *argv])
    opts = ["-t"] if tty else []
    return ssh([*opts, "-o", "ConnectTimeout=20", LOGIN, remote]).returncode


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    p = argparse.ArgumentParser(
        prog="gix",
        description="Run GPU jobs on the GIX cluster from your laptop.",
        epilog="Paths are relative to your NAS folder (gix whoami).",
    )
    p.add_argument("--version", action="version", version=f"gix {__version__}")
    sub = p.add_subparsers(dest="cmd", metavar="<command>")

    s = sub.add_parser("setup", help="one-time setup: SSH key and config (asks for your NetID password once)")
    s.add_argument("--netid")
    s = sub.add_parser("submit", help="submit a job script, e.g. gix submit myproject/train.sbatch")
    s.add_argument("script")
    s.add_argument("sbatch_args", nargs=argparse.REMAINDER, help="extra sbatch options (optional)")
    sub.add_parser("status", help="your queued, running and recently finished jobs")
    s = sub.add_parser("logs", help="show a job's output (-f to follow it live)")
    s.add_argument("jobid")
    s.add_argument("-f", "--follow", action="store_true")
    s = sub.add_parser("cancel", help="cancel a job (or 'all')")
    s.add_argument("jobid")
    s = sub.add_parser("upload", help="copy a file or folder from your laptop to your NAS folder")
    s.add_argument("path", help="file or folder on your laptop, e.g. ./my-project")
    s.add_argument("--to", help="subfolder on the NAS to put it in (default: your NAS folder itself)")
    s = sub.add_parser("download", help="copy a file or folder from your NAS folder to your laptop")
    s.add_argument("path", help="path in your NAS folder, e.g. my-project/outputs")
    s.add_argument("--to", help="folder on your laptop (default: current folder)")
    sub.add_parser("whoami", help="your NetID and NAS folder")

    a = p.parse_args(argv)
    if a.cmd is None:
        p.print_help()
        return 0
    if a.cmd == "setup":
        return cmd_setup(a)
    if not SSH_CONFIG.exists() or LOGIN not in SSH_CONFIG.read_text():
        sys.exit("gix: run `gix setup` first")
    if a.cmd == "submit":
        return forward(["submit", a.script, *a.sbatch_args])
    if a.cmd == "logs":
        return forward(["logs", a.jobid, *(["-f"] if a.follow else [])], tty=a.follow)
    if a.cmd == "upload":
        return cmd_upload(a)
    if a.cmd == "download":
        return cmd_download(a)
    if a.cmd == "cancel":
        return forward(["cancel", a.jobid])
    return forward([a.cmd])


if __name__ == "__main__":
    raise SystemExit(main())
