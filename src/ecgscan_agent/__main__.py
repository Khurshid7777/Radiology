"""The desk agent's command line (SD-152).

    python -m ecgscan_agent setup --server https://cardio.vision-ai.uz --key <the line from /agent> --folder <folder>
    python -m ecgscan_agent run          (keeps watching; started with the computer)
    python -m ecgscan_agent once         (one look at the folders)
    python -m ecgscan_agent status
"""

from __future__ import annotations

import argparse
import logging
import sys

from ecgscan_agent.core import Agent, Memory, Settings, home


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ecgscan_agent", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--home", default=None, help="settings folder (default: the user's app data)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup")
    s.add_argument("--server", required=True)
    s.add_argument("--key", required=True)
    s.add_argument("--folder", action="append", required=True)
    s.add_argument("--ext", action="append", default=None)
    sub.add_parser("run")
    sub.add_parser("once")
    sub.add_parser("status")
    a = ap.parse_args(argv)
    from pathlib import Path

    base = Path(a.home) if a.home else home()
    cfg_path, mem_path = base / "config.json", base / "sent.json"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stderr), logging.FileHandler(base / "agent.log")]
                        if base.is_dir() else None)
    if a.cmd == "setup":
        st = Settings(server=a.server.strip(), key=a.key.strip(), folders=a.folder, **({"exts": a.ext} if a.ext else {}))
        st.check()
        st.save(cfg_path)
        n = Agent(st, Memory(mem_path)).baseline()
        print(f"Настроено. Папок: {len(st.folders)}; уже лежащих записей (не отправляются): {n}.")
        return 0
    if not cfg_path.exists():
        print("Сначала настройка: ecgscan_agent setup --server ... --key ... --folder ...", file=sys.stderr)
        return 2
    st = Settings.load(cfg_path)
    agent = Agent(st, Memory(mem_path))
    if a.cmd == "status":
        done = agent.mem.done
        print(f"Сервер: {st.server}; папок: {len(st.folders)}; отправлено: "
              f"{sum(1 for v in done.values() if v['outcome'] == 'sent')}; отклонено: "
              f"{sum(1 for v in done.values() if v['outcome'].startswith('refused'))}")
        return 0
    if a.cmd == "once":
        print(agent.round())
        return 0
    agent.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
