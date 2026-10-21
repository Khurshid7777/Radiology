"""Telegram client of the ECG scanner (TZ §6.5, SD-009).

``logic`` holds everything testable without Telegram: the whitelist, what to do
with each kind of message, the API calls and the RU/UZ verdict. ``__main__`` is the
aiogram adapter (``python -m ecgscan_bot``), imported only when the bot runs.
Messages and photos are not stored by the bot; logs carry ids and statuses only.
"""
