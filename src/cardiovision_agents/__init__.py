"""CardioVision agents on the server, driven from one Telegram supergroup (D279).

One bot reads the owner's supergroup; each forum topic belongs to one agent (coordinator, developer, security, news,
triage). A message from the owner in a topic starts that agent: Claude Code (headless, the owner's subscription) as an
unprivileged user in its own clone of the repository. Nobody but the owner can start an agent or approve a deploy.
"""
